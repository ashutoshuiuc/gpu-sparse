"""Threshold-compacted top-k selection for chunked scoring.

Motivation. Our component breakdown puts top-k selection at 17.8% of GPU time at
N=1M with B=500 and 37.6% at N=100K with B=1, making it the largest remaining cost
after traversal. In the chunked scorer the selection runs `torch.topk` over every
column of each chunk, but scores are sparse: at MS MARCO scale a query touches about
17% of the documents in a 131,072-document chunk, so selection spends most of its
time scanning zeros.

Method. Selection is run over a compacted candidate set instead of the full chunk.
After the first chunk establishes a running top-k, its k-th best score tau is a valid
lower bound on what can enter the final top-k: any document scoring below tau is
already beaten by k documents, so it cannot appear in the answer. For each later chunk
we therefore compact only the columns with score >= tau and select among those. As tau
rises the candidate count collapses, and later chunks contribute almost nothing.

This is exact, not a pruning heuristic. Every posting is still accumulated in full and
every document is still scored; the threshold only governs which fully computed scores
are handed to the selector. We keep score >= tau rather than > tau so that documents
tied with the current k-th best are retained, which keeps the returned score multiset
identical to unpruned selection.

Overflow is handled rather than assumed away: the candidate buffer is fixed-width, and
any query whose candidate count exceeds it falls back to a full `torch.topk` for that
chunk, so the result never depends on the buffer being large enough.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_above_threshold_kernel(
    scores_ptr,        # [B, chunk_len] float32, row stride = score_stride
    tau_ptr,           # [B] float32: current k-th best per query
    out_idx_ptr,       # [B, max_cand] int32, chunk-local column indices
    out_val_ptr,       # [B, max_cand] float32
    count_ptr,         # [B] int32, pre-zeroed
    score_stride,
    chunk_len,
    max_cand,
    BLOCK: tl.constexpr,
):
    """One program per (query, column block). Reserves output slots with an atomic."""
    q = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)

    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < chunk_len
    vals = tl.load(scores_ptr + q * score_stride + offs, mask=mask, other=float("-inf"))

    tau = tl.load(tau_ptr + q)
    keep = mask & (vals >= tau)
    n_keep = tl.sum(keep.to(tl.int32), axis=0)
    if n_keep == 0:
        return

    # reserve a contiguous run of slots for this block's survivors
    base = tl.atomic_add(count_ptr + q, n_keep)

    # position of each survivor within this block, via an exclusive prefix sum
    k32 = keep.to(tl.int32)
    pos = tl.cumsum(k32, axis=0) - k32
    slot = base + pos
    ok = keep & (slot < max_cand)
    tl.store(out_idx_ptr + q * max_cand + slot, offs.to(tl.int32), mask=ok)
    tl.store(out_val_ptr + q * max_cand + slot, vals, mask=ok)


def compacted_topk(scores, tau, k, max_cand=8192, block=1024):
    """Top-k of `scores` [B, L] considering only entries >= tau[B].

    Returns (values [B,k], indices [B,k]) with chunk-local indices. Rows whose
    candidate count exceeds max_cand fall back to a full topk, so the answer is
    independent of max_cand.
    """
    B, L = scores.shape
    dev = scores.device
    kk = min(k, L)
    max_cand = max(max_cand, kk)

    idx_buf = torch.zeros((B, max_cand), device=dev, dtype=torch.int32)
    val_buf = torch.full((B, max_cand), float("-inf"), device=dev, dtype=torch.float32)
    counts = torch.zeros(B, device=dev, dtype=torch.int32)

    grid = (B, triton.cdiv(L, block))
    _compact_above_threshold_kernel[grid](
        scores, tau, idx_buf, val_buf, counts,
        scores.stride(0), L, max_cand, BLOCK=block,
    )

    v, i = torch.topk(val_buf, k=kk, dim=1)
    idx = torch.gather(idx_buf.to(torch.int64), 1, i)

    # any row that overflowed the buffer is redone exactly
    over = (counts > max_cand)
    if bool(over.any()):
        rows = torch.nonzero(over, as_tuple=True)[0]
        fv, fi = torch.topk(scores[rows], k=kk, dim=1)
        v[rows] = fv
        idx[rows] = fi
    return v, idx


def triton_chunked_score_fused(index, query_term_ids, query_term_scores, top_k=10,
                               doc_chunk=131072, block_pl=128, chunk_bounds=None,
                               max_cand=8192):
    """Chunked scoring with threshold-compacted selection.

    Identical in result to `triton_chunked_score`; differs only in how each chunk's
    candidates reach the selector.
    """
    from triton_kernel_chunked import _chunked_scatter_add_kernel, build_chunk_boundaries

    B, max_qterms = query_term_ids.shape
    N = int(index.num_docs)
    dev = index.device
    k = min(top_k, N)

    if chunk_bounds is None:
        chunk_bounds = build_chunk_boundaries(index, doc_chunk)
    n_chunks = chunk_bounds.shape[0] - 1

    run_s = torch.full((B, k), float("-inf"), device=dev, dtype=torch.float32)
    run_i = torch.zeros((B, k), device=dev, dtype=torch.int64)
    buf = torch.zeros(B, min(doc_chunk, N), device=dev, dtype=torch.float32)

    for c in range(n_chunks):
        c0 = c * doc_chunk
        clen = min(doc_chunk, N - c0)
        view = buf[:, :clen]
        view.zero_()
        _chunked_scatter_add_kernel[(B, max_qterms)](
            index.doc_ids, index.scores,
            chunk_bounds[c], chunk_bounds[c + 1],
            query_term_ids, query_term_scores,
            view, c0, clen, view.stride(0), max_qterms, index.vocab_size,
            BLOCK_PL=block_pl,
        )

        kk = min(k, clen)
        if c == 0:
            # nothing to threshold against yet
            s, i = torch.topk(view, k=kk, dim=1)
        else:
            # run_s is descending, so its last column is the current k-th best
            tau = run_s[:, -1].contiguous()
            s, i = compacted_topk(view, tau, kk, max_cand=max_cand)
        i = i + c0

        if c == 0 and kk == k:
            run_s, run_i = s, i
            continue
        cat_s = torch.cat([run_s, s], dim=1)
        cat_i = torch.cat([run_i, i], dim=1)
        run_s, sel = torch.topk(cat_s, k=k, dim=1)
        run_i = torch.gather(cat_i, 1, sel)

    return run_s, run_i
