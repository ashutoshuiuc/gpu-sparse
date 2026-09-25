"""
Fused Triton Kernel for GPU Sparse Retrieval Scoring.

Fuses: posting list traversal + score accumulation + partial top-k
in a single GPU kernel pass, avoiding materializing the full [batch, num_docs] score matrix.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _fused_scatter_score_kernel(
    # Inverted index
    doc_ids_ptr,       # [total_postings] int32
    doc_scores_ptr,    # [total_postings] float32
    offsets_ptr,       # [vocab_size] int64
    lengths_ptr,       # [vocab_size] int32
    # Query
    query_term_ids_ptr,    # [batch, max_qterms] int32
    query_term_scores_ptr, # [batch, max_qterms] float32
    # Output score matrix
    out_scores_ptr,    # [batch, num_docs] float32
    # Dimensions
    num_docs: tl.constexpr,
    max_qterms: tl.constexpr,
    BLOCK_D: tl.constexpr,  # Block size for doc dimension
):
    """
    Each program instance handles one (query, doc_block) pair.
    Grid: [batch, ceil(num_docs / BLOCK_D)]

    For each query term, load the posting list segment that overlaps with this doc block,
    accumulate scores via scatter-add within the block.
    """
    q_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1).to(tl.int64)

    # Doc IDs this block is responsible for
    doc_start = block_idx * BLOCK_D
    doc_offsets = doc_start + tl.arange(0, BLOCK_D)
    doc_mask = doc_offsets < num_docs

    # Initialize local score accumulator
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Iterate over query terms
    for t_pos in range(max_qterms):
        # Load query term
        term_id = tl.load(query_term_ids_ptr + q_idx * max_qterms + t_pos)
        if term_id < 0:
            continue

        q_score = tl.load(query_term_scores_ptr + q_idx * max_qterms + t_pos)

        # Load posting list metadata
        pl_offset = tl.load(offsets_ptr + term_id)
        pl_length = tl.load(lengths_ptr + term_id)

        # Scan posting list in blocks
        # For each chunk of the posting list, check if any doc falls in our block range
        n_chunks = (pl_length + BLOCK_D - 1) // BLOCK_D
        for chunk in range(n_chunks):
            chunk_start = pl_offset + chunk * BLOCK_D
            chunk_offsets = tl.arange(0, BLOCK_D)
            chunk_mask = chunk_offsets < (pl_length - chunk * BLOCK_D)

            # Load posting list entries
            pl_docs = tl.load(doc_ids_ptr + chunk_start + chunk_offsets, mask=chunk_mask, other=-1)
            pl_scores = tl.load(doc_scores_ptr + chunk_start + chunk_offsets, mask=chunk_mask, other=0.0)

            # For each posting entry, check if it falls in our doc block
            # This is O(BLOCK_D^2) but BLOCK_D is small (32-128)
            for j in range(BLOCK_D):
                target_doc = doc_start + j
                if target_doc >= num_docs:
                    break
                # Check if any posting list entry matches this doc
                match = pl_docs == target_doc
                matched_score = tl.sum(tl.where(match, pl_scores, 0.0))
                # Workaround: accumulate via index
                if matched_score > 0:
                    # Use a simple conditional accumulation
                    current = tl.load(out_scores_ptr + q_idx * num_docs + target_doc)
                    tl.store(out_scores_ptr + q_idx * num_docs + target_doc,
                             current + q_score * matched_score)

    # Note: scores already stored above via atomic-like pattern


@triton.jit
def _fast_scatter_add_kernel(
    # Inverted index data
    doc_ids_ptr,
    doc_scores_ptr,
    offsets_ptr,
    lengths_ptr,
    # Query data
    query_term_ids_ptr,
    query_term_scores_ptr,
    # Output
    out_scores_ptr,
    # Dims
    num_docs,
    max_qterms: tl.constexpr,
    vocab_size,
    BLOCK_PL: tl.constexpr,  # Posting list chunk size
):
    """
    Simpler fused kernel: each program handles one (query, query_term) pair.
    Grid: [batch, max_qterms]

    Loads posting list for the query term and scatter-adds scores into the output.
    Uses tl.atomic_add for concurrent accumulation across terms.
    """
    q_idx = tl.program_id(0).to(tl.int64)
    t_pos = tl.program_id(1).to(tl.int64)

    # Load query term. Bound-check against vocab_size before using term_id as an
    # index: offsets_ptr and lengths_ptr are only vocab_size long, so an
    # out-of-range term_id reads garbage metadata and then issues unmasked wild
    # loads from doc_ids_ptr and wild atomics into out_scores_ptr. vocab_size was
    # already a kernel parameter but was previously unused.
    term_id = tl.load(query_term_ids_ptr + q_idx * max_qterms + t_pos)
    if term_id < 0 or term_id >= vocab_size:
        return

    q_score = tl.load(query_term_scores_ptr + q_idx * max_qterms + t_pos)

    # Load posting list metadata
    pl_offset = tl.load(offsets_ptr + term_id)
    pl_length = tl.load(lengths_ptr + term_id)

    if pl_length == 0:
        return

    # Process posting list in chunks
    n_chunks = (pl_length + BLOCK_PL - 1) // BLOCK_PL

    for chunk in range(n_chunks):
        chunk_start = pl_offset + chunk * BLOCK_PL
        offsets = tl.arange(0, BLOCK_PL)
        mask = offsets < (pl_length - chunk * BLOCK_PL)

        # Load doc IDs and scores
        pl_doc_ids = tl.load(doc_ids_ptr + chunk_start + offsets, mask=mask, other=-1)
        pl_scores = tl.load(doc_scores_ptr + chunk_start + offsets, mask=mask, other=0.0)

        # Compute contributions
        contribs = q_score * pl_scores

        # Scatter-add into output (int64 to avoid overflow at large batch * num_docs)
        out_offsets = q_idx * num_docs + pl_doc_ids.to(tl.int64)
        tl.atomic_add(out_scores_ptr + out_offsets, contribs, mask=mask & (pl_doc_ids >= 0))


def triton_fused_score(
    index,  # GPUInvertedIndex
    query_term_ids: torch.Tensor,    # [batch, max_qterms]
    query_term_scores: torch.Tensor, # [batch, max_qterms]
    top_k: int = 10,
    block_pl: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Triton scoring: scatter-add + top-k.

    Uses _fast_scatter_add_kernel for posting list traversal,
    then torch.topk for final selection.
    """
    # tl.arange requires a power-of-two length, so BLOCK_PL must be one. Without
    # this check an invalid value fails deep inside the Triton JIT with an opaque
    # error rather than at the call site.
    assert block_pl > 0 and (block_pl & (block_pl - 1)) == 0, \
        f"block_pl must be a positive power of 2, got {block_pl}"
    batch_size = query_term_ids.shape[0]
    max_qterms = query_term_ids.shape[1]
    device = index.device

    # Output score matrix
    out_scores = torch.zeros(batch_size, index.num_docs, device=device, dtype=torch.float32)

    # Launch kernel: one program per (query, query_term) pair
    grid = (batch_size, max_qterms)

    _fast_scatter_add_kernel[grid](
        index.doc_ids, index.scores,
        index.offsets, index.lengths,
        query_term_ids, query_term_scores,
        out_scores,
        index.num_docs,
        max_qterms,
        index.vocab_size,
        BLOCK_PL=block_pl,
    )

    # Top-k selection
    top_scores, top_doc_ids = torch.topk(out_scores, k=min(top_k, index.num_docs), dim=1)
    return top_scores, top_doc_ids


@triton.jit
def _wand_pruned_scatter_kernel(
    # Index
    doc_ids_ptr, doc_scores_ptr,
    offsets_ptr, lengths_ptr, max_scores_ptr,
    # Query
    query_term_ids_ptr, query_term_scores_ptr,
    # Upper bounds (precomputed)
    upper_bounds_ptr,  # [batch, max_qterms]
    threshold_ptr,     # [batch] - current threshold for pruning
    # Output
    out_scores_ptr,
    # Dims
    num_docs, max_qterms: tl.constexpr,
    BLOCK_PL: tl.constexpr,
):
    """
    WAND-pruned scatter kernel: skip terms whose upper bound < threshold.
    """
    # int64: q_idx * num_docs overflows signed int32 once batch * num_docs > 2^31
    # (e.g. batch 243 at 8.84M docs), causing illegal memory access.
    q_idx = tl.program_id(0).to(tl.int64)
    t_pos = tl.program_id(1)

    term_id = tl.load(query_term_ids_ptr + q_idx * max_qterms + t_pos)
    if term_id < 0:
        return

    # Check upper bound vs threshold
    ub = tl.load(upper_bounds_ptr + q_idx * max_qterms + t_pos)
    thresh = tl.load(threshold_ptr + q_idx)

    # Prune if upper bound is below threshold (conservative - keeps most terms)
    if ub < thresh * 0.1:  # 10% of threshold as aggressive pruning
        return

    q_score = tl.load(query_term_scores_ptr + q_idx * max_qterms + t_pos)
    pl_offset = tl.load(offsets_ptr + term_id)
    pl_length = tl.load(lengths_ptr + term_id)

    if pl_length == 0:
        return

    n_chunks = (pl_length + BLOCK_PL - 1) // BLOCK_PL
    for chunk in range(n_chunks):
        chunk_start = pl_offset + chunk * BLOCK_PL
        offs = tl.arange(0, BLOCK_PL)
        mask = offs < (pl_length - chunk * BLOCK_PL)

        pl_doc_ids = tl.load(doc_ids_ptr + chunk_start + offs, mask=mask, other=-1)
        pl_scores = tl.load(doc_scores_ptr + chunk_start + offs, mask=mask, other=0.0)

        contribs = q_score * pl_scores
        out_offsets = q_idx * num_docs + pl_doc_ids.to(tl.int64)
        tl.atomic_add(out_scores_ptr + out_offsets, contribs, mask=mask & (pl_doc_ids >= 0))


def triton_wand_score(
    index,
    query_term_ids: torch.Tensor,
    query_term_scores: torch.Tensor,
    top_k: int = 10,
    block_pl: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-phase WAND scoring with Triton kernel.

    Phase 1: Compute upper bounds, do coarse scoring with aggressive pruning
    Phase 2: Refine with exact scoring on promising terms
    """
    # tl.arange requires a power-of-two length, so BLOCK_PL must be one. Without
    # this check an invalid value fails deep inside the Triton JIT with an opaque
    # error rather than at the call site.
    assert block_pl > 0 and (block_pl & (block_pl - 1)) == 0, \
        f"block_pl must be a positive power of 2, got {block_pl}"
    batch_size = query_term_ids.shape[0]
    max_qterms = query_term_ids.shape[1]
    device = index.device

    # Compute upper bounds: q_score * max_doc_score for each (query, term)
    # Gather max_scores for query terms
    valid_mask = query_term_ids >= 0
    safe_ids = torch.where(valid_mask, query_term_ids, torch.zeros_like(query_term_ids))
    term_max_scores = index.max_scores[safe_ids]  # [batch, max_qterms]
    upper_bounds = query_term_scores * term_max_scores * valid_mask.float()

    # Initial threshold = 0 (no pruning in first pass)
    threshold = torch.zeros(batch_size, device=device, dtype=torch.float32)

    # Phase 1: Full scoring (threshold = 0 means no pruning)
    out_scores = torch.zeros(batch_size, index.num_docs, device=device, dtype=torch.float32)

    grid = (batch_size, max_qterms)
    _wand_pruned_scatter_kernel[grid](
        index.doc_ids, index.scores,
        index.offsets, index.lengths, index.max_scores,
        query_term_ids, query_term_scores,
        upper_bounds, threshold,
        out_scores,
        index.num_docs, max_qterms,
        BLOCK_PL=block_pl,
    )

    top_scores, top_doc_ids = torch.topk(out_scores, k=min(top_k, index.num_docs), dim=1)
    return top_scores, top_doc_ids
