"""
Chunked scatter-add scoring with running top-k: O(B*chunk) memory instead of O(B*N).

Motivation. The monolithic path allocates a dense [B, N] fp32 score accumulator,
which is 16.5 GiB at B=500 on MS MARCO's 8.8M passages and grows linearly in both
dimensions. That buffer, not index size, is what bounds batch size: our index is
8.5 GB but the accumulator at B=1000 would be 33 GiB. It is also why concurrent
work (AgentIR) reports being unable to run MS MARCO 8.8M on a 16 GB budget at all.

Design. Partition the document space into chunks of `doc_chunk` documents. For each
chunk we scatter-add only the postings whose doc_id falls inside it, select the
chunk's top-k, and merge into a running top-k. Peak score memory becomes
O(B * doc_chunk + B * k) rather than O(B * N).

The efficiency of this rests on being able to find, for each (term, chunk) pair,
the sub-range of the term's posting list lying in that chunk, so we do not re-read
whole posting lists per chunk. That requires posting lists sorted ascending by
doc_id. The builders previously used a non-stable argsort which left 96.2% of
posting lists unsorted; with that fixed we can precompute the boundaries once with
a per-term searchsorted, costing vocab_size * (n_chunks+1) int64 entries
(about 21 MB at 30,522 terms and 88 chunks on 8.8M documents).

Total posting traffic is unchanged from the monolithic kernel: every posting is
still read exactly once across all chunks. What changes is the accumulator
footprint and the cost of selection, which now runs on a small buffer.
"""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _chunked_scatter_add_kernel(
    doc_ids_ptr,            # [total_postings] int32, ascending within each term
    doc_scores_ptr,         # [total_postings] float32
    chunk_lo_ptr,           # [vocab_size] int64: first posting index in this chunk
    chunk_hi_ptr,           # [vocab_size] int64: one past the last
    query_term_ids_ptr,     # [batch, max_qterms] int32
    query_term_scores_ptr,  # [batch, max_qterms] float32
    out_ptr,                # [batch, chunk_len] float32, pre-zeroed
    chunk_start,            # int: global doc id of column 0
    chunk_len,              # int: number of columns actually used
    out_stride,             # int: row stride of out_ptr, which is NOT chunk_len on
                            #      the final (short) chunk when the buffer is reused
    max_qterms: tl.constexpr,
    vocab_size,
    BLOCK_PL: tl.constexpr,
):
    """One program per (query, query-term). Writes only into this chunk's columns."""
    # int64 so q_idx * chunk_len cannot overflow signed int32
    q_idx = tl.program_id(0).to(tl.int64)
    t_pos = tl.program_id(1).to(tl.int64)

    term_id = tl.load(query_term_ids_ptr + q_idx * max_qterms + t_pos)
    if term_id < 0 or term_id >= vocab_size:
        return

    lo = tl.load(chunk_lo_ptr + term_id)
    hi = tl.load(chunk_hi_ptr + term_id)
    n = hi - lo
    if n <= 0:
        return

    q_score = tl.load(query_term_scores_ptr + q_idx * max_qterms + t_pos)
    base_out = q_idx * out_stride.to(tl.int64)

    n_blocks = (n + BLOCK_PL - 1) // BLOCK_PL
    for b in range(n_blocks):
        offs = tl.arange(0, BLOCK_PL)
        mask = offs < (n - b * BLOCK_PL)
        p = lo + b * BLOCK_PL + offs
        docs = tl.load(doc_ids_ptr + p, mask=mask, other=-1)
        scs = tl.load(doc_scores_ptr + p, mask=mask, other=0.0)
        # translate to chunk-local columns; the boundary search guarantees these
        # are in range, and we mask defensively regardless
        local = docs.to(tl.int64) - chunk_start
        ok = mask & (docs >= 0) & (local >= 0) & (local < chunk_len)
        tl.atomic_add(out_ptr + base_out + local, q_score * scs, mask=ok)


def build_chunk_boundaries(index, doc_chunk: int):
    """Precompute per-(term, chunk) posting sub-ranges. Requires ascending doc_ids.

    Returns an int64 tensor of shape [n_chunks + 1, vocab_size] holding, for each
    chunk boundary, the posting index at which that chunk begins for each term.
    """
    doc_ids = index.doc_ids.cpu().numpy()
    offsets = index.offsets.cpu().numpy().astype(np.int64)
    lengths = index.lengths.cpu().numpy().astype(np.int64)
    V = int(index.vocab_size)
    N = int(index.num_docs)
    n_chunks = (N + doc_chunk - 1) // doc_chunk

    # The per-chunk sub-ranges below come from a binary search, which is only valid
    # if posting lists ascend in doc_id. An unsorted list yields wrong sub-ranges and
    # therefore silently wrong scores, so check rather than trust the builder.
    n_bad = 0
    for t in np.nonzero(lengths)[0]:
        seg = doc_ids[offsets[t]: offsets[t] + lengths[t]]
        if np.any(np.diff(seg) < 0):
            n_bad += 1
    if n_bad:
        raise ValueError(
            f"{n_bad} of {int((lengths > 0).sum())} posting lists are not ascending in "
            f"doc_id. Chunked scoring requires ascending lists (build the index with "
            f"np.argsort(..., kind='stable')); otherwise results are silently wrong.")

    bounds = np.zeros((n_chunks + 1, V), dtype=np.int64)
    edges = [c * doc_chunk for c in range(n_chunks)] + [N]
    for t in range(V):
        L = lengths[t]
        if L == 0:
            bounds[:, t] = offsets[t]
            continue
        seg = doc_ids[offsets[t]: offsets[t] + L]
        # padding entries are -1 and sort first; skip them
        first = int(np.searchsorted(seg, 0, side="left"))
        rel = np.searchsorted(seg[first:], edges, side="left")
        bounds[:, t] = offsets[t] + first + rel
    return torch.from_numpy(bounds).to(index.device)


def triton_chunked_score(index, query_term_ids, query_term_scores, top_k=10,
                         doc_chunk=131072, block_pl=128, chunk_bounds=None):
    """Chunked scoring with running top-k. Peak score memory is O(B*doc_chunk + B*k)."""
    assert block_pl > 0 and (block_pl & (block_pl - 1)) == 0, \
        f"block_pl must be a positive power of 2, got {block_pl}"
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
        # buf is allocated once at width doc_chunk and sliced, so on the final short
        # chunk view.stride(0) is doc_chunk, not clen. The kernel must be told the
        # real stride: assuming clen silently scatters into the wrong rows and
        # corrupts exactly the last chunk (an 8.25%-of-corpus error at N=1M).
        _chunked_scatter_add_kernel[(B, max_qterms)](
            index.doc_ids, index.scores,
            chunk_bounds[c], chunk_bounds[c + 1],
            query_term_ids, query_term_scores,
            view, c0, clen, view.stride(0), max_qterms, index.vocab_size,
            BLOCK_PL=block_pl,
        )
        kk = min(k, clen)
        s, i = torch.topk(view, k=kk, dim=1)
        i = i + c0
        if c == 0 and kk == k:
            run_s, run_i = s, i
            continue
        cat_s = torch.cat([run_s, s], dim=1)
        cat_i = torch.cat([run_i, i], dim=1)
        run_s, sel = torch.topk(cat_s, k=k, dim=1)
        run_i = torch.gather(cat_i, 1, sel)

    return run_s, run_i
