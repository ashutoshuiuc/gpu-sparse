"""Threshold-compacted selection against plain per-chunk topk, on the full 8.84M corpus.

Reports correctness against the monolithic path, end-to-end latency, and how the
candidate count decays across chunks, which is the mechanism the method relies on.
"""
import os
import json, sys, time
import numpy as np, scipy.sparse as sp, torch

GS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{GS}/src"); sys.path.insert(0, GS)
OUT = os.path.join(os.environ.get("GPUSPARSE_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")), "results_fused_topk.json")
CACHE = f"{GS}/data_cache"
DEV = "cuda:0"
res = {}


def t_ms(fn, warmup=3, iters=9):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2]


print("loading 8.8M corpus...", flush=True)
d = np.load(f"{CACHE}/msmarco_8m_coo_v2.npz")
rows, cols, vals = d["rows"], d["cols"], d["vals"]
with open(f"{CACHE}/msmarco_8m_docids_v2.json") as f:
    n_docs = len(json.load(f))
csr = sp.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(n_docs, 30522))
del rows, cols, vals, d

from triton_kernel import triton_fused_score
from triton_kernel_chunked import triton_chunked_score, build_chunk_boundaries
from fused_topk import triton_chunked_score_fused, compacted_topk
from run_correctness_verification import build_gpu_index_from_csr, prepare_queries_from_meta

idx = build_gpu_index_from_csr(csr, DEV); del csr
meta = torch.load(f"{CACHE}/msmarco_splade_csr_1000000_meta.pt", map_location="cpu", weights_only=False)
qi_all, qs_all, _ = prepare_queries_from_meta(meta, max_terms=128, device=DEV)
C, K = 131072, 1000
cb = build_chunk_boundaries(idx, C)
print(f"  {n_docs:,} docs, {cb.shape[0]-1} chunks, {qi_all.shape[0]} queries", flush=True)

# ------------------------------------------------ correctness + latency
print("\n" + "="*72); print("CORRECTNESS AND LATENCY"); print("="*72, flush=True)
rows_out = {}
for B in (16, 64, 256):
    qi, qs = qi_all[:B].contiguous(), qs_all[:B].contiguous()
    s_m, i_m = triton_fused_score(idx, qi, qs, top_k=K)
    s_c, i_c = triton_chunked_score(idx, qi, qs, top_k=K, chunk_bounds=cb, doc_chunk=C)
    s_f, i_f = triton_chunked_score_fused(idx, qi, qs, top_k=K, chunk_bounds=cb, doc_chunk=C)
    torch.cuda.synchronize()

    def ov(a, b, kk):
        return float(np.mean([len(set(a[r, :kk].tolist()) & set(b[r, :kk].tolist()))/kk
                              for r in range(B)]))
    o1000 = ov(i_m, i_f, 1000); o10 = ov(i_m, i_f, 10)
    sd = float((s_m.sort(dim=1, descending=True).values
                - s_f.sort(dim=1, descending=True).values).abs().max())

    t_mono = t_ms(lambda: triton_fused_score(idx, qi, qs, top_k=K))
    t_chk = t_ms(lambda: triton_chunked_score(idx, qi, qs, top_k=K, chunk_bounds=cb, doc_chunk=C))
    t_fus = t_ms(lambda: triton_chunked_score_fused(idx, qi, qs, top_k=K, chunk_bounds=cb, doc_chunk=C))

    print(f"\n-- B={B} --")
    print(f"   vs monolithic: top-1000 overlap {o1000:.6f}, top-10 {o10:.6f}, max score diff {sd:.3e}")
    print(f"   monolithic        {t_mono:8.2f} ms")
    print(f"   chunked           {t_chk:8.2f} ms")
    print(f"   chunked + compact {t_fus:8.2f} ms   -> {t_chk/t_fus:.2f}x vs chunked, "
          f"{t_mono/t_fus:.2f}x vs monolithic", flush=True)
    rows_out[f"B{B}"] = {"top1000_overlap": o1000, "top10_overlap": o10, "max_score_diff": sd,
                         "mono_ms": t_mono, "chunked_ms": t_chk, "fused_ms": t_fus,
                         "speedup_vs_chunked": t_chk/t_fus, "speedup_vs_mono": t_mono/t_fus}
    del s_m, i_m, s_c, i_c, s_f, i_f; torch.cuda.empty_cache()
res["latency"] = rows_out

# ------------------------------------------------ candidate decay
print("\n" + "="*72); print("CANDIDATE COUNT PER CHUNK (the mechanism)"); print("="*72, flush=True)
B = 64
qi, qs = qi_all[:B].contiguous(), qs_all[:B].contiguous()
from triton_kernel_chunked import _chunked_scatter_add_kernel
run_s = torch.full((B, K), float("-inf"), device=DEV)
buf = torch.zeros(B, C, device=DEV, dtype=torch.float32)
decay = []
for c in range(cb.shape[0] - 1):
    c0 = c * C; clen = min(C, n_docs - c0)
    view = buf[:, :clen]; view.zero_()
    _chunked_scatter_add_kernel[(B, qi.shape[1])](
        idx.doc_ids, idx.scores, cb[c], cb[c+1], qi, qs,
        view, c0, clen, view.stride(0), qi.shape[1], idx.vocab_size, BLOCK_PL=128)
    nz = int((view > 0).sum(dim=1).float().mean().item())
    if c == 0:
        s, i = torch.topk(view, k=K, dim=1)
        cand = clen
    else:
        tau = run_s[:, -1].contiguous()
        cand = int((view >= tau.unsqueeze(1)).sum(dim=1).float().mean().item())
        s, i = compacted_topk(view, tau, K)
    i = i + c0
    if c == 0:
        run_s, run_i = s, i
    else:
        cs = torch.cat([run_s, s], 1); ci = torch.cat([run_i, i], 1)
        run_s, sel = torch.topk(cs, k=K, dim=1); run_i = torch.gather(ci, 1, sel)
    decay.append({"chunk": c, "nonzero_cols": nz, "candidates": cand})
    if c < 6 or c % 16 == 0:
        print(f"   chunk {c:3d}: nonzero {nz:7,}  candidates passed to selector {cand:7,} "
              f"({cand/clen:6.2%} of chunk)", flush=True)
mean_after = np.mean([r["candidates"] for r in decay[1:]])
print(f"\n   mean candidates after chunk 0: {mean_after:,.0f} of {C:,} columns "
      f"({mean_after/C:.2%})", flush=True)
res["decay"] = {"per_chunk": decay, "mean_candidates_after_first": float(mean_after),
                "chunk_len": C}

json.dump(res, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}", flush=True)
