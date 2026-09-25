"""Is the chunked/monolithic disagreement on BEIR a chunking artifact or float nondeterminism?

Scatter-add accumulates with tl.atomic_add, so the summation order for a document's score
depends on how blocks happen to interleave and is not reproducible run to run. When two
documents' scores fall within that noise, any selector can order them either way.

The decisive test is self-consistency: run the SAME kernel twice on the SAME inputs and
compare. If the monolithic path already disagrees with itself at the same rate the chunked
path disagrees with it, the disagreement is inherent to atomic accumulation and is not
caused by chunking. If the monolithic path is perfectly self-consistent while the chunked
path differs, chunking is responsible and the claim of identical rankings is wrong.

We also count how many documents sit within 1e-5 of the k-th best score, which is the
population that can reorder at all.
"""
import os
import json, sys
import numpy as np, scipy.sparse as sp, torch

GS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BW = os.environ.get("GPUSPARSE_DATA_BEIR13", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/beir13"))
sys.path.insert(0, f"{GS}/src"); sys.path.insert(0, GS)
OUT = os.path.join(os.environ.get("GPUSPARSE_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")), "results_selfconsistency.json")

from triton_kernel import triton_fused_score
from triton_kernel_chunked import triton_chunked_score, build_chunk_boundaries
from fused_topk import triton_chunked_score_fused
from run_correctness_verification import build_gpu_index_from_csr

DEV, C, K = "cuda:0", 131072, 1000
res = {}


def ov(a, b, n, B):
    return float(np.mean([len(set(a[r, :n].tolist()) & set(b[r, :n].tolist()))/n
                          for r in range(B)]))


for key in ("beir_trec-covid", "beir_quora_test"):
    csr = sp.load_npz(f"{BW}/cache_v2/{key}_docs.npz").tocsr().astype(np.float32)
    meta = torch.load(f"{BW}/cache_v2/{key}_meta.pt", map_location="cpu", weights_only=False)
    q_csr = meta["query_csr"].tocsr().astype(np.float32)
    nq = q_csr.shape[0]; mx = int(np.diff(q_csr.indptr).max())
    ti = np.full((nq, mx), -1, dtype=np.int32); ts = np.zeros((nq, mx), dtype=np.float32)
    for i in range(nq):
        a, b = q_csr.indptr[i], q_csr.indptr[i+1]
        ti[i, :b-a] = q_csr.indices[a:b]; ts[i, :b-a] = q_csr.data[a:b]
    B = min(200, nq)
    qi = torch.from_numpy(ti[:B]).to(DEV); qs = torch.from_numpy(ts[:B]).to(DEV)

    idx = build_gpu_index_from_csr(csr, DEV)
    cb = build_chunk_boundaries(idx, C)
    kk = min(K, csr.shape[0])

    # same kernel, twice
    m1_s, m1_i = triton_fused_score(idx, qi, qs, top_k=kk)
    m2_s, m2_i = triton_fused_score(idx, qi, qs, top_k=kk)
    c1_s, c1_i = triton_chunked_score(idx, qi, qs, top_k=kk, chunk_bounds=cb, doc_chunk=C)
    c2_s, c2_i = triton_chunked_score(idx, qi, qs, top_k=kk, chunk_bounds=cb, doc_chunk=C)
    f1_s, f1_i = triton_chunked_score_fused(idx, qi, qs, top_k=kk, chunk_bounds=cb, doc_chunk=C)
    torch.cuda.synchronize()

    # how many documents sit within fp32 noise of the k-th best score
    tau = m1_s[:, -1:]
    near = int(((m1_s - tau).abs() < 1e-5).sum(dim=1).float().mean().item())
    # bitwise identity of the score vectors across two identical runs
    bitwise_same = bool(torch.equal(m1_s, m2_s))

    ent = {
        "docs": int(csr.shape[0]), "queries": B,
        "mono_vs_mono_top10": ov(m1_i, m2_i, 10, B),
        "mono_vs_mono_top1000": ov(m1_i, m2_i, kk, B),
        "mono_scores_bitwise_identical": bitwise_same,
        "mono_vs_mono_max_score_diff": float((m1_s - m2_s).abs().max()),
        "chunked_vs_chunked_top10": ov(c1_i, c2_i, 10, B),
        "chunked_vs_chunked_top1000": ov(c1_i, c2_i, kk, B),
        "mono_vs_chunked_top10": ov(m1_i, c1_i, 10, B),
        "mono_vs_chunked_top1000": ov(m1_i, c1_i, kk, B),
        "mono_vs_compact_top10": ov(m1_i, f1_i, 10, B),
        "mono_vs_compact_top1000": ov(m1_i, f1_i, kk, B),
        "docs_within_1e-5_of_kth": near,
    }
    res[key] = ent
    print(f"\n{key}: {ent['docs']:,} docs, {B} queries, k={kk}")
    print(f"   monolithic scores bitwise identical across two runs: {bitwise_same}")
    print(f"   monolithic vs itself   top-10 {ent['mono_vs_mono_top10']:.6f}  "
          f"top-{kk} {ent['mono_vs_mono_top1000']:.6f}  max diff {ent['mono_vs_mono_max_score_diff']:.3e}")
    print(f"   chunked   vs itself    top-10 {ent['chunked_vs_chunked_top10']:.6f}  "
          f"top-{kk} {ent['chunked_vs_chunked_top1000']:.6f}")
    print(f"   monolithic vs chunked  top-10 {ent['mono_vs_chunked_top10']:.6f}  "
          f"top-{kk} {ent['mono_vs_chunked_top1000']:.6f}")
    print(f"   monolithic vs compact  top-10 {ent['mono_vs_compact_top10']:.6f}  "
          f"top-{kk} {ent['mono_vs_compact_top1000']:.6f}")
    print(f"   docs within 1e-5 of the k-th best score: {near} per query", flush=True)
    del idx, cb; torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
print("\nVERDICT: if 'monolithic vs itself' is below 1.0 at a comparable rate, the")
print("disagreement is atomic-accumulation nondeterminism and not caused by chunking.")
