"""Verify chunked and threshold-compacted scoring on BEIR corpora, not just MS MARCO.

The lsr-benchmark engine now routes any collection above one chunk through the chunked
path, so exactness needs checking on data whose sparsity differs from MS MARCO. Quora
(522,931 docs) and Touche (382,545) are the cached BEIR corpora larger than one chunk;
TREC-COVID (171,332) is just over the boundary and included as a third point.
"""
import os
import json, sys
import numpy as np, scipy.sparse as sp, torch

GS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BW = os.environ.get("GPUSPARSE_DATA_BEIR13", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/beir13"))
sys.path.insert(0, f"{GS}/src"); sys.path.insert(0, GS)
OUT = os.path.join(os.environ.get("GPUSPARSE_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")), "results_chunked_beir.json")

from triton_kernel import triton_fused_score
from triton_kernel_chunked import triton_chunked_score, build_chunk_boundaries
from fused_topk import triton_chunked_score_fused
from run_correctness_verification import build_gpu_index_from_csr

DEV, C, K = "cuda:0", 131072, 1000
res = {}

for key in ("beir_trec-covid", "beir_webis-touche2020_v2", "beir_quora_test"):
    csr = sp.load_npz(f"{BW}/cache_v2/{key}_docs.npz").tocsr().astype(np.float32)
    meta = torch.load(f"{BW}/cache_v2/{key}_meta.pt", map_location="cpu", weights_only=False)
    q_csr = meta["query_csr"].tocsr().astype(np.float32)

    # dense [nq, max_nnz] query tensors, untruncated
    nq = q_csr.shape[0]
    mx = int(np.diff(q_csr.indptr).max())
    ti = np.full((nq, mx), -1, dtype=np.int32); ts = np.zeros((nq, mx), dtype=np.float32)
    for i in range(nq):
        a, b = q_csr.indptr[i], q_csr.indptr[i+1]
        ti[i, :b-a] = q_csr.indices[a:b]; ts[i, :b-a] = q_csr.data[a:b]
    B = min(200, nq)
    qi = torch.from_numpy(ti[:B]).to(DEV); qs = torch.from_numpy(ts[:B]).to(DEV)

    idx = build_gpu_index_from_csr(csr, DEV)
    cb = build_chunk_boundaries(idx, C)
    kk = min(K, csr.shape[0])

    s_m, i_m = triton_fused_score(idx, qi, qs, top_k=kk)
    s_c, i_c = triton_chunked_score(idx, qi, qs, top_k=kk, chunk_bounds=cb, doc_chunk=C)
    s_f, i_f = triton_chunked_score_fused(idx, qi, qs, top_k=kk, chunk_bounds=cb, doc_chunk=C)
    torch.cuda.synchronize()

    def ov(a, b, n):
        return float(np.mean([len(set(a[r, :n].tolist()) & set(b[r, :n].tolist()))/n
                              for r in range(B)]))

    def sd(a, b):
        return float((a.sort(dim=1, descending=True).values
                      - b.sort(dim=1, descending=True).values).abs().max())

    ent = {"docs": int(csr.shape[0]), "queries_scored": B, "chunks": cb.shape[0]-1,
           "nnz_per_doc": float(csr.nnz/csr.shape[0]),
           "chunked_top10": ov(i_m, i_c, 10), "chunked_top1000": ov(i_m, i_c, kk),
           "chunked_max_diff": sd(s_m, s_c),
           "compact_top10": ov(i_m, i_f, 10), "compact_top1000": ov(i_m, i_f, kk),
           "compact_max_diff": sd(s_m, s_f)}
    res[key] = ent
    print(f"{key}: {ent['docs']:,} docs, {ent['chunks']} chunks, "
          f"{ent['nnz_per_doc']:.1f} terms/doc, {B} queries")
    print(f"   chunked          top-10 {ent['chunked_top10']:.6f}  top-{kk} "
          f"{ent['chunked_top1000']:.6f}  max diff {ent['chunked_max_diff']:.3e}")
    print(f"   chunked+compact  top-10 {ent['compact_top10']:.6f}  top-{kk} "
          f"{ent['compact_top1000']:.6f}  max diff {ent['compact_max_diff']:.3e}", flush=True)
    del idx, cb, s_m, i_m, s_c, i_c, s_f, i_f; torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=2)
bad = [k for k, v in res.items()
       if min(v["chunked_top1000"], v["compact_top1000"], v["chunked_top10"], v["compact_top10"]) < 1.0]
print(f"\nwrote {OUT}")
print("ALL EXACT" if not bad else f"MISMATCH on: {bad}")
