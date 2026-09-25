"""Chunked vs monolithic scoring on the full 8.84M-passage MS MARCO corpus.

This is the measurement the chunked kernel exists for. The monolithic path allocates a
dense [B, N] fp32 accumulator, so at N=8.84M it needs 33.8 MiB per query in the batch
regardless of how sparse the query is. That, not the 8.5 GB index, is what caps batch
size. Chunked scoring replaces it with [B, doc_chunk] plus a running top-k.

We report three things:
  1. correctness: top-k overlap and max score difference against the monolithic path,
     at a batch size both can run
  2. peak memory and latency at a common batch size
  3. the largest batch size each path can actually reach on one 80 GB H100, which is
     the operational difference
"""
import json, os, sys, time
import numpy as np, scipy.sparse as sp, torch

GS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{GS}/src"); sys.path.insert(0, GS)
OUT = os.path.join(os.environ.get("GPUSPARSE_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")), "results_8m.json")
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


print("loading cached 8.8M COO...", flush=True)
d = np.load(f"{CACHE}/msmarco_8m_coo_v2.npz")
rows, cols, vals = d["rows"], d["cols"], d["vals"]
with open(f"{CACHE}/msmarco_8m_docids_v2.json") as f:
    n_docs = len(json.load(f))
V = 30522
print(f"  {n_docs:,} docs, {len(rows):,} postings, avg {len(rows)/n_docs:.1f} terms/doc", flush=True)

csr = sp.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(n_docs, V))
del rows, cols, vals, d

from triton_kernel import triton_fused_score
from triton_kernel_chunked import triton_chunked_score, build_chunk_boundaries
from run_correctness_verification import build_gpu_index_from_csr, prepare_queries_from_meta

print("building GPU index...", flush=True)
idx = build_gpu_index_from_csr(csr, DEV)
idx_gib = (idx.doc_ids.numel() * 4 + idx.scores.numel() * 4) / 2**30
print(f"  index resident: {idx_gib:.2f} GiB", flush=True)
del csr

# queries: reuse the cached 1M-subset query encodings (same query set, 6980 dev queries)
meta = torch.load(f"{CACHE}/msmarco_splade_csr_1000000_meta.pt", map_location="cpu", weights_only=False)
qi_all, qs_all, _ = prepare_queries_from_meta(meta, max_terms=128, device=DEV)
print(f"  {qi_all.shape[0]} queries, max_terms=128", flush=True)

DOC_CHUNK = 131072
t0 = time.time(); cb = build_chunk_boundaries(idx, DOC_CHUNK); bt = time.time() - t0
n_chunks = cb.shape[0] - 1
print(f"  chunk boundaries: {n_chunks} chunks, {cb.numel()*8/2**20:.1f} MiB, built in {bt:.1f}s "
      f"(validated all posting lists ascending)", flush=True)
res["setup"] = {"n_docs": n_docs, "index_gib": idx_gib, "n_chunks": n_chunks,
                "boundary_mib": cb.numel()*8/2**20, "boundary_build_s": bt}

# ---------------------------------------------------------------- 1. correctness
print("\n" + "="*72); print("1. CORRECTNESS at B=64 (a batch the monolithic path can hold)")
print("="*72, flush=True)
B0, K = 64, 1000
qi, qs = qi_all[:B0].contiguous(), qs_all[:B0].contiguous()
s_m, i_m = triton_fused_score(idx, qi, qs, top_k=K)
s_c, i_c = triton_chunked_score(idx, qi, qs, top_k=K, chunk_bounds=cb, doc_chunk=DOC_CHUNK)
torch.cuda.synchronize()
ov = float(np.mean([len(set(i_m[r, :K].tolist()) & set(i_c[r, :K].tolist()))/K for r in range(B0)]))
sd = float((s_m.sort(dim=1, descending=True).values - s_c.sort(dim=1, descending=True).values).abs().max())
ov10 = float(np.mean([len(set(i_m[r, :10].tolist()) & set(i_c[r, :10].tolist()))/10 for r in range(B0)]))
print(f"   top-1000 overlap {ov:.6f} | top-10 overlap {ov10:.6f} | max score diff {sd:.3e}", flush=True)
res["correctness"] = {"B": B0, "top1000_overlap": ov, "top10_overlap": ov10, "max_score_diff": sd}
del s_m, i_m, s_c, i_c; torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2. memory + latency
print("\n" + "="*72); print("2. PEAK MEMORY AND LATENCY"); print("="*72, flush=True)
rows_out = {}
for B in (16, 64, 256):
    qi, qs = qi_all[:B].contiguous(), qs_all[:B].contiguous()
    acc = B * n_docs * 4 / 2**30
    ent = {"accumulator_gib_monolithic": acc,
           "accumulator_gib_chunked": B * min(DOC_CHUNK, n_docs) * 4 / 2**30}
    for tag, fn in (("mono", lambda: triton_fused_score(idx, qi, qs, top_k=K)),
                    ("chunked", lambda: triton_chunked_score(idx, qi, qs, top_k=K,
                                                             chunk_bounds=cb, doc_chunk=DOC_CHUNK))):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t = t_ms(fn); m = torch.cuda.max_memory_allocated()/2**30
            ent[f"{tag}_ms"] = t; ent[f"{tag}_peak_gib"] = m
            print(f"   B={B:4d} {tag:8s} {t:9.2f} ms  peak {m:6.2f} GiB "
                  f"(accumulator alone {ent['accumulator_gib_'+('monolithic' if tag=='mono' else 'chunked')]:.2f} GiB)",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            ent[f"{tag}_ms"] = None; print(f"   B={B:4d} {tag:8s} OOM", flush=True)
            torch.cuda.empty_cache()
    if ent.get("mono_ms") and ent.get("chunked_ms"):
        print(f"        -> memory {ent['mono_peak_gib']/ent['chunked_peak_gib']:.2f}x lower, "
              f"latency {ent['chunked_ms']/ent['mono_ms']:.2f}x", flush=True)
    rows_out[f"B{B}"] = ent
res["scaling"] = rows_out

# ---------------------------------------------------------------- 3. max batch size
print("\n" + "="*72); print("3. LARGEST BATCH EACH PATH REACHES ON ONE 80 GB H100")
print("="*72, flush=True)
# The cached query set has only 500 queries, so reaching batch sizes that actually
# exhaust an 80 GB H100 requires replicating them. That is sound for this test: the
# accumulator footprint depends only on B and N, not on query content. Replicated
# queries are used ONLY here, never for any retrieval-quality number.
qrep = torch.cat([qi_all] * 128, dim=0)
srep = torch.cat([qs_all] * 128, dim=0)
print(f"   (queries replicated to {qrep.shape[0]} for the memory stress test only)", flush=True)
lim = {}
for tag in ("mono", "chunked"):
    best = 0
    for B in (16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        if B > qrep.shape[0]: break
        qi, qs = qrep[:B].contiguous(), srep[:B].contiguous()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            if tag == "mono":
                s, i = triton_fused_score(idx, qi, qs, top_k=K)
            else:
                s, i = triton_chunked_score(idx, qi, qs, top_k=K, chunk_bounds=cb, doc_chunk=DOC_CHUNK)
            torch.cuda.synchronize(); del s, i
            best = B
        except torch.cuda.OutOfMemoryError:
            print(f"   {tag:8s} B={B}: OOM (accumulator would be "
                  f"{B*n_docs*4/2**30:.1f} GiB)" if tag == "mono" else
                  f"   {tag:8s} B={B}: OOM", flush=True)
            torch.cuda.empty_cache(); break
    lim[tag] = best
    print(f"   {tag:8s} max batch reached: {best}", flush=True)
res["max_batch"] = lim
if lim["mono"]:
    print(f"   -> chunked reaches {lim['chunked']/max(lim['mono'],1):.0f}x the batch size", flush=True)

json.dump(res, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}", flush=True)
