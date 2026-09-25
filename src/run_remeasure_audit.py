"""
GPUSparse audit re-measurement (fixes 3 flagged tables).

(1) tab:correctness  -- GPU times must be monotone; original included one-time
    Triton JIT compilation at the first (100K) scale. Fix: warm up kernel before
    timing, time with CUDA events, report GPU(s) + speedup vs CPU dense matmul.
(2) tab:comparison   -- exact methods (GPUSparse Triton, dense matmul, cuSPARSE)
    must produce IDENTICAL MRR@10/nDCG over the same data. Fix: run all three with
    the SAME query tensors (same max_terms, same top-k) and report MRR for each.
(3) tab:multigpu     -- single-GPU baseline must be labeled with the batch size it
    was measured at. Fix: measure single-GPU and 2-GPU split at one stated batch.

All inputs are the cached MS MARCO SPLADE CSR (100K/500K/1M). No estimates.
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
    build_gpu_index_from_csr, prepare_queries_from_meta, compute_recall_exact,
)

CACHE = BASE / "data_cache"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
DEVICE = "cuda:0"
TOPK = 1000


def cuda_time(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1000.0  # seconds


def load(num_docs):
    csr = sp.load_npz(CACHE / f"msmarco_splade_csr_{num_docs}.npz")
    meta = torch.load(CACHE / f"msmarco_splade_csr_{num_docs}_meta.pt",
                      map_location="cpu", weights_only=False)
    return csr, meta


def fix1_correctness(scales=(100000, 500000, 1000000)):
    """Warmup-timed GPU correctness vs CPU dense matmul ground truth."""
    out = {}
    for num_docs in scales:
        if not (CACHE / f"msmarco_splade_csr_{num_docs}.npz").exists():
            print(f"  skip {num_docs}: no cache"); continue
        csr, meta = load(num_docs)
        q_ids, q_scores, q_dense = prepare_queries_from_meta(meta, max_terms=128, device=DEVICE)
        idx = build_gpu_index_from_csr(csr, DEVICE)

        # GPU: warm up (JIT compile) THEN time
        fn = lambda: triton_fused_score(idx, q_ids, q_scores, top_k=TOPK)
        gpu_s = cuda_time(fn, warmup=5, iters=20)
        g_sc, g_ids = fn()

        # CPU dense matmul ground truth
        dd = torch.from_numpy(csr.toarray().astype(np.float32))
        qd = torch.from_numpy(q_dense.astype(np.float32))
        t0 = time.time()
        cpu_sc = qd @ dd.t()
        _, c_ids = torch.topk(cpu_sc, k=TOPK, dim=1)
        cpu_s = time.time() - t0

        rec = compute_recall_exact(g_ids.cpu(), c_ids, k_values=[10, 100, 1000])
        out[str(num_docs)] = {
            "gpu_s": round(gpu_s, 4), "cpu_s": round(cpu_s, 3),
            "speedup": round(cpu_s / gpu_s, 1),
            "recall": rec,
        }
        print(f"  {num_docs}: GPU={gpu_s*1000:.2f}ms CPU={cpu_s:.2f}s speedup={cpu_s/gpu_s:.0f}x R@1000={rec.get('recall@1000'):.4f}")
        del dd, cpu_sc, idx; torch.cuda.empty_cache()
    return out


def fix2_comparison_mrr(num_docs=100000):
    """All exact methods, IDENTICAL queries/top-k -> MRR must match.
    Loads qrels from meta if present; else reports ranking agreement only."""
    csr, meta = load(num_docs)
    q_ids, q_scores, q_dense = prepare_queries_from_meta(meta, max_terms=128, device=DEVICE)
    idx = build_gpu_index_from_csr(csr, DEVICE)

    # GPUSparse Triton
    g_sc, g_ids = triton_fused_score(idx, q_ids, q_scores, top_k=TOPK)
    g_ids = g_ids.cpu()

    # Dense matmul (GPU, exact)
    dd = torch.from_numpy(csr.toarray().astype(np.float32)).to(DEVICE)
    qd = torch.from_numpy(q_dense.astype(np.float32)).to(DEVICE)
    d_sc = qd @ dd.t()
    _, d_ids = torch.topk(d_sc, k=TOPK, dim=1)
    d_ids = d_ids.cpu()

    # cuSPARSE / torch.sparse.mm (exact)
    q_sp = torch.from_numpy(q_dense.astype(np.float32)).to_sparse_csr().to(DEVICE)
    sp_sc = torch.sparse.mm(q_sp, dd.t())
    _, s_ids = torch.topk(sp_sc, k=TOPK, dim=1)
    s_ids = s_ids.cpu()

    # Ranking agreement between the three exact methods (should be ~identical)
    def overlap(a, b, k):
        return float(np.mean([len(set(a[i, :k].tolist()) & set(b[i, :k].tolist())) / k
                              for i in range(a.shape[0])]))
    res = {
        "num_docs": num_docs,
        "triton_vs_dense_overlap@10": overlap(g_ids, d_ids, 10),
        "triton_vs_dense_overlap@1000": overlap(g_ids, d_ids, 1000),
        "dense_vs_cusparse_overlap@10": overlap(d_ids, s_ids, 10),
        "dense_vs_cusparse_overlap@1000": overlap(d_ids, s_ids, 1000),
    }
    # If qrels are available in meta, compute MRR for each method identically.
    #
    # BUG FIX: this previously looked up qrels.get(qi) with qi a 0-based batch
    # index, but qrels is keyed by MS MARCO query-id STRINGS. Every lookup missed,
    # n stayed 0, and the function returned 0.0 for all three methods. Because
    # the check was meant to verify that exact methods produce identical MRR, its
    # silent failure let a spurious +/-0.004 "spread" stand in the paper. We now
    # map batch position -> qid via meta["qids"] and assert we matched something.
    qrels = meta.get("qrels") if isinstance(meta, dict) else None
    qid_list = meta.get("qids") if isinstance(meta, dict) else None
    if qrels and qid_list is None:
        print("  WARNING: qrels present but meta['qids'] missing; cannot map batch "
              "index -> query id, so MRR is not computable. Skipping.")
        qrels = None
    if qrels:
        def mrr10(ids):
            s = 0.0; n = 0
            for qi in range(ids.shape[0]):
                qid = str(qid_list[qi])
                rel = qrels.get(qid) or qrels.get(qid_list[qi]) or {}
                if not rel: continue
                n += 1
                for r, did in enumerate(ids[qi, :10].tolist()):
                    if rel.get(did, 0) > 0 or rel.get(str(did), 0) > 0:
                        s += 1.0 / (r + 1); break
            if n == 0:
                raise RuntimeError(
                    "mrr10 matched 0 queries against qrels; the qid mapping is wrong. "
                    "Refusing to report 0.0 as a result.")
            return s / n if n else None
        res["mrr@10_triton"] = mrr10(g_ids)
        res["mrr@10_dense"] = mrr10(d_ids)
        res["mrr@10_cusparse"] = mrr10(s_ids)
    else:
        res["note"] = "qrels not in meta; reported ranking agreement (exact methods should be ~1.0)"
    print(f"  comparison@{num_docs}: {json.dumps(res)}")
    del dd, qd, d_sc, sp_sc, idx; torch.cuda.empty_cache()
    return res


def fix3_multigpu(num_docs=100000, batch=500):
    """Single-GPU at a STATED batch, plus 2-GPU split (if 2 GPUs)."""
    csr, meta = load(num_docs)
    q_ids, q_scores, _ = prepare_queries_from_meta(meta, max_terms=128, device=DEVICE)
    q_ids, q_scores = q_ids[:batch], q_scores[:batch]
    idx = build_gpu_index_from_csr(csr, DEVICE)
    one = cuda_time(lambda: triton_fused_score(idx, q_ids, q_scores, top_k=TOPK), iters=20)
    res = {"num_docs": num_docs, "batch": batch, "single_gpu_ms": round(one * 1000, 3),
           "n_gpus_available": torch.cuda.device_count()}
    if torch.cuda.device_count() >= 2:
        half = batch // 2
        idx1 = build_gpu_index_from_csr(csr, "cuda:1")
        qa_i, qa_s = q_ids[:half], q_scores[:half]
        qb_i = q_ids[half:].to("cuda:1"); qb_s = q_scores[half:].to("cuda:1")
        def two():
            triton_fused_score(idx, qa_i, qa_s, top_k=TOPK)
            triton_fused_score(idx1, qb_i, qb_s, top_k=TOPK)
            torch.cuda.synchronize("cuda:0"); torch.cuda.synchronize("cuda:1")
        two_ms = cuda_time(two, iters=20) * 1000
        res["two_gpu_ms"] = round(two_ms, 3)
        res["scaling_factor"] = round(one * 1000 / two_ms, 2)
    print(f"  multigpu: {json.dumps(res)}")
    return res


def main():
    print("GPU:", torch.cuda.get_device_name(0), "| n_gpus:", torch.cuda.device_count())
    out = {}
    print("\n[FIX 1] correctness (warmup-timed)")
    out["correctness"] = fix1_correctness()
    print("\n[FIX 2] comparison MRR (identical queries)")
    out["comparison"] = fix2_comparison_mrr(100000)
    print("\n[FIX 3] multi-GPU (stated batch)")
    out["multigpu"] = fix3_multigpu(100000, batch=500)
    p = RESULTS / "remeasure_audit.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
