"""
Triton Kernel V4: Document-Parallel Scoring with CSR Format.

Key insight: Instead of (query, term) -> scatter to random doc positions,
use (query, doc) -> gather from doc's term list.

This eliminates ALL atomic operations by giving each program exclusive
ownership of a document. Writes are perfectly coalesced.

Uses a document-centric CSR index:
- For each document d, store (term_id, term_score) pairs
- Each program handles one (query, doc) pair
- For each doc term, look up query weight in dense query matrix
"""

import torch
import triton
import triton.language as tl
from typing import Tuple
import numpy as np
try:
    import scipy.sparse as sp
except ImportError:  # scipy is optional; the guard below degrades gracefully
    sp = None
import time


@triton.jit
def _doc_csr_score_kernel(
    # Document CSR data (doc -> terms)
    doc_term_ids_ptr,    # [total_doc_terms] int32
    doc_term_scores_ptr, # [total_doc_terms] float32
    doc_offsets_ptr,     # [num_docs] int64
    doc_lengths_ptr,     # [num_docs] int32
    # Query lookup table: dense [batch, vocab_size] float32
    query_weights_ptr,
    # Output
    out_scores_ptr,      # [batch, num_docs] float32
    # Dims
    num_docs,
    vocab_size: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,  # Upper bound on chunks per doc
    BLOCK_T: tl.constexpr,     # Terms per chunk
):
    """
    One program per (query, doc). Grid: [batch, num_docs].

    Process document terms in chunks, looking up query weights.
    Zero atomics. Coalesced writes.
    """
    # int64: q_idx * num_docs + doc_id overflows signed int32 once
    # batch * num_docs > 2^31, causing illegal memory access.
    doc_id = tl.program_id(0).to(tl.int64)
    q_idx = tl.program_id(1).to(tl.int64)

    # Load document metadata
    d_off = tl.load(doc_offsets_ptr + doc_id)
    d_len = tl.load(doc_lengths_ptr + doc_id)

    # Pointer to this query's weight row
    q_weight_row_ptr = query_weights_ptr + q_idx * vocab_size

    # Accumulate score in a 1-element tensor; we'll extract the scalar at the end
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Fixed loop bound (constexpr), with early-exit via mask
    for chunk in range(MAX_CHUNKS):
        chunk_off = chunk * BLOCK_T
        offsets = tl.arange(0, BLOCK_T)
        remaining = d_len - chunk_off
        mask = offsets < remaining

        # Load doc terms and scores
        load_offsets = d_off + chunk_off + offsets
        term_ids = tl.load(doc_term_ids_ptr + load_offsets, mask=mask, other=0)
        doc_scores = tl.load(doc_term_scores_ptr + load_offsets, mask=mask, other=0.0)

        # Look up query weights (gathered reads from dense query matrix)
        q_addrs = q_weight_row_ptr + term_ids.to(tl.int64)
        q_weights = tl.load(q_addrs, mask=mask, other=0.0)

        # Accumulate dot product contribution
        acc += doc_scores * q_weights * mask

    # Write final score (reduce acc to scalar)
    score = tl.sum(acc, axis=0)
    out_offset = q_idx * num_docs + doc_id
    # Store scalar to scalar pointer
    tl.store(out_scores_ptr + out_offset, score)


def build_doc_csr_index(doc_reps, device):
    """
    Build a document-centric CSR index from dense SPLADE representations.

    For each document, store (term_id, term_score) pairs sorted by term_id.

    IMPORTANT: doc_reps must be a DENSE array (or torch.Tensor) of shape
    [num_docs, vocab_size]. This is a real scalability limit of the
    document-parallel path: the dense form is num_docs * 30522 * 4 bytes, i.e.
    about 12.2 GB at 100K documents and 122 GB at 1M, which is why the
    document-parallel kernel is only evaluated up to 500K documents.

    Passing a scipy sparse matrix used to fail catastrophically rather than
    clearly: `doc_np[rows, cols]` returns a (1, nnz) np.matrix, and the
    subsequent `vals[sort_idx]` then performs matrix fancy-indexing and attempts
    an (nnz, nnz) allocation (528 TiB at 100K documents). We now reject that
    input explicitly.
    """
    t0 = time.time()
    if sp is not None and sp.issparse(doc_reps):
        raise TypeError(
            "build_doc_csr_index expects a dense [num_docs, vocab_size] array, "
            "got a scipy sparse matrix. Densify explicitly if you have the "
            f"memory ({doc_reps.shape[0] * doc_reps.shape[1] * 4 / 2**30:.1f} GiB "
            "for this input), or use the scatter-add path, which consumes the "
            "sparse inverted index directly."
        )
    doc_np = doc_reps.numpy() if isinstance(doc_reps, torch.Tensor) else doc_reps
    doc_np = np.asarray(doc_np)          # np.matrix -> ndarray, so indexing is 1-D
    if doc_np.ndim != 2:
        raise ValueError(f"expected a 2-D [num_docs, vocab_size] array, got shape {doc_np.shape}")
    num_docs, vocab_size = doc_np.shape

    # Get non-zero entries
    rows, cols = np.nonzero(doc_np)
    vals = np.asarray(doc_np[rows, cols]).ravel().astype(np.float32)

    # Sort by (doc_id, term_id) for CSR format
    sort_idx = np.lexsort((cols, rows))
    sorted_docs = rows[sort_idx]
    sorted_terms = cols[sort_idx].astype(np.int32)
    sorted_scores = vals[sort_idx]

    # Build CSR offsets
    doc_counts = np.bincount(sorted_docs, minlength=num_docs).astype(np.int32)
    doc_offsets = np.zeros(num_docs, dtype=np.int64)
    if num_docs > 1:
        np.cumsum(doc_counts[:-1], out=doc_offsets[1:])

    total_entries = len(sorted_terms)
    max_doc_len = int(doc_counts.max())
    elapsed = time.time() - t0
    mem_mb = (total_entries * 8 + num_docs * 12) / 1e6
    print(f"  Doc-CSR index: {num_docs} docs, {total_entries} entries, "
          f"max_doc_len={max_doc_len}, {mem_mb:.0f} MB, built in {elapsed:.1f}s")

    return {
        'doc_term_ids': torch.from_numpy(sorted_terms).to(device),
        'doc_term_scores': torch.from_numpy(sorted_scores).to(device),
        'doc_offsets': torch.from_numpy(doc_offsets).to(device),
        'doc_lengths': torch.from_numpy(doc_counts).to(device),
        'num_docs': num_docs,
        'vocab_size': vocab_size,
        'total_entries': total_entries,
        'max_doc_len': max_doc_len,
        'device': device,
    }


def build_query_weight_matrix(query_reps, device):
    """
    Build dense query weight matrix [num_queries, vocab_size] on GPU.

    Memory: 500 queries * 30522 vocab * 4 bytes = 58 MB.
    """
    if isinstance(query_reps, torch.Tensor):
        return query_reps.to(device=device, dtype=torch.float32)
    else:
        return torch.from_numpy(query_reps).to(device=device, dtype=torch.float32)


def triton_doc_csr_score(
    doc_csr_index: dict,
    query_weights: torch.Tensor,  # [batch, vocab_size] dense on GPU
    top_k: int = 10,
    block_t: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    V4 document-CSR scoring.

    One program per (query, doc). No atomics.
    """
    batch_size = query_weights.shape[0]
    vocab_size = query_weights.shape[1]
    num_docs = doc_csr_index['num_docs']
    device = doc_csr_index['device']
    max_doc_len = doc_csr_index['max_doc_len']

    # Compute max chunks needed
    max_chunks = (max_doc_len + block_t - 1) // block_t

    out_scores = torch.zeros(batch_size, num_docs, device=device, dtype=torch.float32)

    grid = (num_docs, batch_size)

    _doc_csr_score_kernel[grid](
        doc_csr_index['doc_term_ids'],
        doc_csr_index['doc_term_scores'],
        doc_csr_index['doc_offsets'],
        doc_csr_index['doc_lengths'],
        query_weights,
        out_scores,
        num_docs,
        vocab_size,
        MAX_CHUNKS=max_chunks,
        BLOCK_T=block_t,
    )

    top_scores, top_doc_ids = torch.topk(out_scores, k=min(top_k, num_docs), dim=1)
    return top_scores, top_doc_ids


def triton_doc_csr_score_chunked(
    doc_csr_index: dict,
    query_weights: torch.Tensor,
    top_k: int = 10,
    block_t: int = 128,
    query_chunk_size: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked version for large query batches.
    """
    batch_size = query_weights.shape[0]
    num_docs = doc_csr_index['num_docs']

    all_top_scores = []
    all_top_doc_ids = []

    for start in range(0, batch_size, query_chunk_size):
        end = min(start + query_chunk_size, batch_size)
        chunk_weights = query_weights[start:end]
        chunk_scores, chunk_ids = triton_doc_csr_score(
            doc_csr_index, chunk_weights, top_k=top_k, block_t=block_t,
        )
        all_top_scores.append(chunk_scores)
        all_top_doc_ids.append(chunk_ids)

    return torch.cat(all_top_scores, dim=0), torch.cat(all_top_doc_ids, dim=0)
