"""
GPUSparse fused Triton kernel vs SPARe's `iterative` scatter-add mode.

SPARe's iterative path (verified in github.com/ieeta-pt/SPARe,
spare/backend_torch.py): store the collection as a CSC tensor (column/term-major
== inverted index); for each query term, slice that term's posting list
(ccol[t]:ccol[t+1] -> row_indices, values) and accumulate document scores with
torch's index_add_ (a scatter-add); then torch.topk. We reimplement that path
faithfully (the published SPARe code is a PyTorch index_add_ loop) and time it
against our single fused Triton kernel on identical 100K/500K/1M SPLADE data.

This isolates GPUSparse's actual delta over SPARe: the fused kernel + warp-aligned
layout, NOT the scatter-add reformulation (which SPARe's iterative mode shares).
No estimates; both timed with CUDA events on the same GPU.
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse as sp

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.triton_kernel import triton_fused_score
from src.run_correctness_verification import (
    build_gpu_index_from_csr, prepare_queries_from_meta,
)

CACHE = BASE / "data_cache"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
DEVICE = "cuda:0"
TOPK = 1000


def cuda_time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1000.0  # seconds


def build_spare_csc(csr_mat, device):
    """SPARe iterative layout: CSC (term-major inverted index) on GPU."""
    csc = csr_mat.tocsc()
    ccol = torch.from_numpy(csc.indptr.astype(np.int64)).to(device)      # [vocab+1]
    rindices = torch.from_numpy(csc.indices.astype(np.int64)).to(device) # doc ids
    cvalues = torch.from_numpy(csc.data.astype(np.float32)).to(device)
    return ccol, rindices, cvalues, csc.shape[0]


def spare_iterative_batch(ccol, rindices, cvalues, num_docs, q_ids, q_scores, top_k):
    """Faithful reimpl of SPARe CSRSparseRetrievalModelIterative.forward,
    looped over a query batch (SPARe processes queries independently)."""
    B = q_ids.shape[0]
    out_scores = torch.empty(B, top_k, device=ccol.device)
    out_ids = torch.empty(B, top_k, dtype=torch.long, device=ccol.device)
    for b in range(B):
        acc = torch.zeros(num_docs, dtype=torch.float32, device=ccol.device)
        ids = q_ids[b]; sc = q_scores[b]
        valid = ids >= 0
        for t in range(ids.shape[0]):
            if not bool(valid[t]):
                continue
            term = int(ids[t]); w = sc[t]
            s, e = int(ccol[term]), int(ccol[term + 1])
            if e > s:
                acc.index_add_(0, rindices[s:e], cvalues[s:e] * w)
        vals, idx = torch.topk(acc, k=top_k, dim=0)
        out_scores[b] = vals; out_ids[b] = idx
    return out_scores, out_ids


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    scales = [100000, 500000, 1000000]
    BATCH = 500
    results = {}
    for num_docs in scales:
        p = CACHE / f"msmarco_splade_csr_{num_docs}.npz"
        if not p.exists():
            print(f"skip {num_docs}: no cache"); continue
        csr = sp.load_npz(p)
        meta = torch.load(CACHE / f"msmarco_splade_csr_{num_docs}_meta.pt",
                          map_location="cpu", weights_only=False)
        q_ids, q_scores, _ = prepare_queries_from_meta(meta, max_terms=128, device=DEVICE)
        q_ids, q_scores = q_ids[:BATCH], q_scores[:BATCH]

        # Ours: fused Triton kernel
        idx = build_gpu_index_from_csr(csr, DEVICE)
        ours = cuda_time(lambda: triton_fused_score(idx, q_ids, q_scores, top_k=TOPK))

        # SPARe iterative (PyTorch index_add_ scatter-add over CSC posting lists)
        ccol, rind, cval, nd = build_spare_csc(csr, DEVICE)
        spare = cuda_time(lambda: spare_iterative_batch(ccol, rind, cval, nd, q_ids, q_scores, TOPK),
                          warmup=2, iters=5)

        # correctness: do they agree on top-10?
        _, oid = triton_fused_score(idx, q_ids, q_scores, top_k=TOPK)
        _, sid = spare_iterative_batch(ccol, rind, cval, nd, q_ids, q_scores, TOPK)
        ov = float(np.mean([len(set(oid[i, :10].tolist()) & set(sid[i, :10].tolist())) / 10
                            for i in range(BATCH)]))

        r = {"num_docs": num_docs, "batch": BATCH,
             "ours_triton_ms": round(ours * 1000, 3),
             "spare_iterative_ms": round(spare * 1000, 3),
             "speedup_ours_over_spare": round(spare / ours, 2),
             "top10_overlap": round(ov, 4)}
        results[str(num_docs)] = r
        print(f"  {num_docs}: ours={r['ours_triton_ms']}ms  SPARe-iter={r['spare_iterative_ms']}ms  "
              f"{r['speedup_ours_over_spare']}x  overlap@10={ov:.3f}")
        del idx, ccol, rind, cval; torch.cuda.empty_cache()

    outp = RESULTS / "spare_iterative_comparison.json"
    json.dump(results, open(outp, "w"), indent=2)
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
