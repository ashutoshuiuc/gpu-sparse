"""
GPU-Friendly Inverted Index Data Structure for Learned Sparse Retrieval.

Design:
- Posting lists stored as contiguous GPU tensors (doc_ids, scores)
- Block-aligned: each posting list padded to multiple of BLOCK_SIZE (warp size = 32)
- Sorted by doc_id within each posting list for merge-join efficiency
- Metadata: offsets, lengths, max_scores per posting list for WAND-style pruning
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import time


BLOCK_SIZE = 32  # Warp size for coalesced access


@dataclass
class GPUInvertedIndex:
    """GPU-resident inverted index with block-aligned posting lists."""
    # Flattened posting lists (all terms concatenated)
    doc_ids: torch.Tensor      # [total_postings] int32 - document IDs
    scores: torch.Tensor       # [total_postings] float32 - term weights
    # Per-term metadata
    offsets: torch.Tensor      # [vocab_size] int64 - start offset in flattened array
    lengths: torch.Tensor      # [vocab_size] int32 - actual length (before padding)
    padded_lengths: torch.Tensor  # [vocab_size] int32 - padded length (multiple of BLOCK_SIZE)
    max_scores: torch.Tensor   # [vocab_size] float32 - max score per posting list (for WAND)
    # Index metadata
    num_docs: int
    vocab_size: int
    device: torch.device


def build_gpu_inverted_index(
    doc_term_ids: list,    # list of lists: doc_term_ids[i] = term IDs for doc i
    doc_term_scores: list, # list of lists: doc_term_scores[i] = scores for doc i
    vocab_size: int,
    device: torch.device = torch.device("cuda:0"),
) -> GPUInvertedIndex:
    """
    Build a GPU-friendly inverted index from document sparse representations.

    Args:
        doc_term_ids: For each document, list of non-zero term IDs
        doc_term_scores: For each document, corresponding term scores
        vocab_size: Size of the vocabulary
        device: Target GPU device

    Returns:
        GPUInvertedIndex on the specified device
    """
    num_docs = len(doc_term_ids)

    # Step 1: Build CPU posting lists (term -> [(doc_id, score), ...])
    posting_lists = [[] for _ in range(vocab_size)]
    for doc_id in range(num_docs):
        for term_id, score in zip(doc_term_ids[doc_id], doc_term_scores[doc_id]):
            posting_lists[term_id].append((doc_id, score))

    # Step 2: Sort each posting list by doc_id
    for term_id in range(vocab_size):
        posting_lists[term_id].sort(key=lambda x: x[0])

    # Step 3: Compute padded lengths and offsets
    lengths = np.zeros(vocab_size, dtype=np.int32)
    padded_lengths = np.zeros(vocab_size, dtype=np.int32)
    max_scores_np = np.zeros(vocab_size, dtype=np.float32)
    offsets = np.zeros(vocab_size, dtype=np.int64)

    for term_id in range(vocab_size):
        pl = posting_lists[term_id]
        lengths[term_id] = len(pl)
        # Pad to multiple of BLOCK_SIZE
        padded_len = ((len(pl) + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        padded_lengths[term_id] = padded_len
        if len(pl) > 0:
            max_scores_np[term_id] = max(s for _, s in pl)

    # Compute offsets (prefix sum of padded lengths)
    offsets[0] = 0
    for i in range(1, vocab_size):
        offsets[i] = offsets[i - 1] + padded_lengths[i - 1]

    total_padded = int(offsets[-1] + padded_lengths[-1]) if vocab_size > 0 else 0

    # Step 4: Fill flattened arrays
    all_doc_ids = np.full(total_padded, -1, dtype=np.int32)  # -1 = padding
    all_scores = np.zeros(total_padded, dtype=np.float32)

    for term_id in range(vocab_size):
        pl = posting_lists[term_id]
        off = offsets[term_id]
        for j, (doc_id, score) in enumerate(pl):
            all_doc_ids[off + j] = doc_id
            all_scores[off + j] = score

    # Step 5: Transfer to GPU
    index = GPUInvertedIndex(
        doc_ids=torch.from_numpy(all_doc_ids).to(device),
        scores=torch.from_numpy(all_scores).to(device),
        offsets=torch.from_numpy(offsets).to(device),
        lengths=torch.from_numpy(lengths).to(device),
        padded_lengths=torch.from_numpy(padded_lengths).to(device),
        max_scores=torch.from_numpy(max_scores_np).to(device),
        num_docs=num_docs,
        vocab_size=vocab_size,
        device=device,
    )
    return index


def build_gpu_inverted_index_from_sparse(
    sparse_matrix: torch.Tensor,  # [num_docs, vocab_size] sparse or dense
    device: torch.device = torch.device("cuda:0"),
) -> GPUInvertedIndex:
    """Build index from a sparse matrix (convenience wrapper)."""
    if sparse_matrix.is_sparse:
        sparse_matrix = sparse_matrix.to_dense()

    sparse_np = sparse_matrix.cpu().numpy()
    num_docs, vocab_size = sparse_np.shape

    doc_term_ids = []
    doc_term_scores = []
    for i in range(num_docs):
        nz = np.nonzero(sparse_np[i])[0]
        doc_term_ids.append(nz.tolist())
        doc_term_scores.append(sparse_np[i, nz].tolist())

    return build_gpu_inverted_index(doc_term_ids, doc_term_scores, vocab_size, device)


def generate_synthetic_splade_data(
    num_docs: int,
    vocab_size: int = 30522,
    avg_terms_per_doc: int = 150,
    score_range: Tuple[float, float] = (0.1, 3.0),
    zipf_a: float = 1.5,
    seed: int = 42,
) -> Tuple[list, list]:
    """
    Generate synthetic SPLADE-like sparse representations.

    Term distribution follows Zipf's law (realistic for natural language).
    Scores drawn uniformly from score_range (approximation of SPLADE log1p(ReLU(x))).
    """
    rng = np.random.RandomState(seed)

    # Zipfian term probabilities
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, zipf_a)
    probs /= probs.sum()

    doc_term_ids = []
    doc_term_scores = []

    for _ in range(num_docs):
        # Number of terms per doc: Poisson around avg
        n_terms = max(10, rng.poisson(avg_terms_per_doc))
        n_terms = min(n_terms, vocab_size)

        # Sample terms from Zipfian distribution (no replacement)
        terms = rng.choice(vocab_size, size=n_terms, replace=False, p=probs)
        # Scores: uniform in range, roughly modeling SPLADE outputs
        scores = rng.uniform(score_range[0], score_range[1], size=n_terms).astype(np.float32)

        doc_term_ids.append(terms.tolist())
        doc_term_scores.append(scores.tolist())

    return doc_term_ids, doc_term_scores


def index_memory_stats(index: GPUInvertedIndex) -> dict:
    """Report memory usage of the GPU index."""
    doc_ids_bytes = index.doc_ids.nelement() * index.doc_ids.element_size()
    scores_bytes = index.scores.nelement() * index.scores.element_size()
    meta_bytes = (
        index.offsets.nelement() * index.offsets.element_size()
        + index.lengths.nelement() * index.lengths.element_size()
        + index.padded_lengths.nelement() * index.padded_lengths.element_size()
        + index.max_scores.nelement() * index.max_scores.element_size()
    )
    total = doc_ids_bytes + scores_bytes + meta_bytes
    return {
        "doc_ids_MB": doc_ids_bytes / 1e6,
        "scores_MB": scores_bytes / 1e6,
        "metadata_MB": meta_bytes / 1e6,
        "total_MB": total / 1e6,
        "num_postings": int(index.lengths.sum().item()),
        "num_padded": int(index.padded_lengths.sum().item()),
        "padding_overhead": 1.0 - int(index.lengths.sum().item()) / max(1, int(index.padded_lengths.sum().item())),
    }
