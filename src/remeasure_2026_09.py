"""Re-measurement for the September 2026 audit of the GPUSparse paper.

Four things the audit could not settle from stored artifacts:

1. `quality`. MRR@10 at 100K appears as 0.892 in Table tab:quality and 0.888 in
   Table tab:comparison for what looks like the same configuration, and
   `results/remeasure_audit.json` records `mrr@10_triton = 0.0` because that run had no
   qrels loaded. Recompute MRR@10, nDCG@10 and Recall@1000 at 100K, 500K and 1M with
   official qrels and trec_eval gain semantics, so one number survives.

2. `exactness`. The abstract claims Recall@10, @100 and @1000 all equal 1.000 with the
   full query vector, while section 9 and the conclusion say >= 0.999 and Table
   tab:correctness shows 0.9988 to 0.9996. The stored sweep resolves this (the 0.999
   figures are the `max_terms=64` truncated runs and 128 is exact), but only at 100K on
   one run. Re-run the sweep at all three scales, twice, so the claim is stated at the
   right strength.

3. `selfconsistency`. The exactness claim is bounded by the kernel's own run-to-run
   variation, since scatter-add accumulates through atomic_add. Re-measure the bound on
   MS MARCO rather than only on BEIR.

4. `invariant`. The paper says the original builder used NumPy's default quicksort and
   left 96.2% of posting lists non-ascending, and that a stable sort plus validation was
   added. Check the invariant directly on a built index.

Separately, `ncu_remeasure_2026_09.sh` collects the doc-parallel kernel's counters at
B=500. The paper reports 48.9 GB "measured" at B=500, but the stored counter was taken at
B=50 and multiplied by ten.

Latency is CUDA-event timed. Counters come from ncu only.
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)
CACHE = f"{REPO}/data_cache"
DEVICE = "cuda:0"


def t_ms(fn, warmup=5, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


# ---------------------------------------------------------------- data

def load_scale(n_docs):
    csr = sp.load_npz(f"{CACHE}/msmarco_splade_csr_{n_docs}.npz").tocsr().astype(np.float32)
    meta = torch.load(f"{CACHE}/msmarco_splade_csr_{n_docs}_meta.pt",
                      map_location="cpu", weights_only=False)
    return csr, meta


def queries_from_meta(meta, max_terms=None):
    """Build the (n_queries, max_terms) padded query tensors.

    max_terms=None sizes to the longest query present, which is what the paper's
    exactness claim requires. A fixed width silently drops the lowest-weighted terms of
    any longer query.
    """
    qd = meta["query_dense"]
    if isinstance(qd, torch.Tensor):
        qd = qd.numpy()
    nq = qd.shape[0]
    nnz = [np.nonzero(qd[i])[0] for i in range(nq)]
    longest = max(len(x) for x in nnz)
    width = longest if max_terms is None else max_terms
    ti = np.full((nq, width), -1, dtype=np.int32)
    ts = np.zeros((nq, width), dtype=np.float32)
    for i, idx in enumerate(nnz):
        vals = qd[i, idx]
        order = np.argsort(-vals)[:width]
        k = len(order)
        ti[i, :k] = idx[order]
        ts[i, :k] = vals[order]
    stats = {"n_queries": nq, "mean_nnz": float(np.mean([len(x) for x in nnz])),
             "max_nnz": int(longest), "width_used": int(width),
             "n_truncated": int(sum(len(x) > width for x in nnz))}
    return (torch.from_numpy(ti).to(DEVICE), torch.from_numpy(ts).to(DEVICE), qd, stats)


# ---------------------------------------------------------------- metrics

def mrr_at_k(ranked_ids, qids, qrels, k=10):
    tot = 0.0
    for i, qid in enumerate(qids):
        rel = qrels.get(str(qid), qrels.get(qid, {})) or {}
        rel = {str(d) for d, g in rel.items() if float(g) > 0}
        if not rel:
            continue
        for r, did in enumerate(ranked_ids[i][:k]):
            if str(did) in rel:
                tot += 1.0 / (r + 1)
                break
    return tot / len(qids)


def ndcg_at_k(ranked_ids, qids, qrels, k=10):
    """trec_eval ndcg_cut semantics: gain is the relevance value itself."""
    out = 0.0
    for i, qid in enumerate(qids):
        rel = qrels.get(str(qid), qrels.get(qid, {})) or {}
        rel = {str(d): float(g) for d, g in rel.items() if float(g) > 0}
        if not rel:
            continue
        dcg = sum(rel.get(str(d), 0.0) / np.log2(r + 2)
                  for r, d in enumerate(ranked_ids[i][:k]))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        if idcg > 0:
            out += dcg / idcg
    return out / len(qids)


def recall_at_k(ranked_ids, qids, qrels, k=1000):
    out = 0.0
    for i, qid in enumerate(qids):
        rel = qrels.get(str(qid), qrels.get(qid, {})) or {}
        rel = {str(d) for d, g in rel.items() if float(g) > 0}
        if not rel:
            continue
        hit = len(rel & {str(d) for d in ranked_ids[i][:k]})
        out += hit / len(rel)
    return out / len(qids)


def set_overlap(a, b, k):
    return float(np.mean([len(set(a[r, :k].tolist()) & set(b[r, :k].tolist())) / k
                          for r in range(a.shape[0])]))


# ---------------------------------------------------------------- runs

def run_quality_and_exactness(scales, repeats=2):
    from triton_kernel import triton_fused_score
    from run_correctness_verification import build_gpu_index_from_csr

    out = {"_why": "settle the 0.892-vs-0.888 MRR discrepancy and the 1.000-vs-0.999 "
                   "recall discrepancy with one measurement per scale.",
           "scales": {}}

    for n_docs in scales:
        ent = {}
        try:
            csr, meta = load_scale(n_docs)
            qids = meta["qids"]
            qrels = meta["qrels"]
            # doc ids: the subset rows are 0..N-1; map back if the meta carries ids
            docids = meta.get("docids")
            idx = build_gpu_index_from_csr(csr, DEVICE)
            k = min(1000, csr.shape[0])

            # exactness sweep: fixed-width truncation versus the full query vector
            sweep = {}
            for mt in (64, 96, 128, None):
                qi, qs, qd, qstats = queries_from_meta(meta, max_terms=mt)
                gpu_s, gpu_i = triton_fused_score(idx, qi, qs, top_k=k)
                # exhaustive dense reference on GPU, in fp32, chunked over documents
                ref_i = dense_reference_topk(csr, qd, k)
                sweep[str(mt)] = {
                    "query_stats": qstats,
                    "recall@10": set_overlap(gpu_i.cpu().numpy(), ref_i, 10),
                    "recall@100": set_overlap(gpu_i.cpu().numpy(), ref_i, 100),
                    "recall@1000": set_overlap(gpu_i.cpu().numpy(), ref_i, min(1000, k)),
                }
                if mt is None:
                    ent["full_query_top1000"] = gpu_i.cpu().numpy()
                del gpu_s, gpu_i, ref_i
                torch.cuda.empty_cache()
            ent["exactness_sweep"] = sweep

            # quality with official qrels, full query vector
            qi, qs, qd, qstats = queries_from_meta(meta, max_terms=None)
            gpu_s, gpu_i = triton_fused_score(idx, qi, qs, top_k=k)
            ranked = gpu_i.cpu().numpy()
            if docids is not None:
                ranked_ids = [[docids[j] for j in row] for row in ranked]
            else:
                ranked_ids = ranked.tolist()
            ent["quality"] = {
                "_note": ("docids mapping present" if docids is not None
                          else "no docids in meta; row index used as document id"),
                "mrr@10": mrr_at_k(ranked_ids, qids, qrels, 10),
                "ndcg@10": ndcg_at_k(ranked_ids, qids, qrels, 10),
                "recall@1000": recall_at_k(ranked_ids, qids, qrels, min(1000, k)),
                "n_queries": len(qids),
            }
            ent["latency_ms"] = t_ms(lambda: triton_fused_score(idx, qi, qs, top_k=k))
            ent["per_query_us"] = ent["latency_ms"] * 1e3 / len(qids)

            # self-consistency: the same kernel twice on identical inputs
            runs = []
            for _ in range(repeats):
                s, i2 = triton_fused_score(idx, qi, qs, top_k=k)
                runs.append((s.clone(), i2.clone()))
            ent["selfconsistency"] = {
                "bitwise_equal_scores": bool(torch.equal(runs[0][0], runs[-1][0])),
                "max_abs_score_diff": float((runs[0][0] - runs[-1][0]).abs().max()),
                "top10_agreement": set_overlap(runs[0][1].cpu().numpy(),
                                               runs[-1][1].cpu().numpy(), 10),
                "top1000_agreement": set_overlap(runs[0][1].cpu().numpy(),
                                                 runs[-1][1].cpu().numpy(), min(1000, k)),
                "docs_within_1e-5_of_kth": float(
                    ((runs[0][0] - runs[0][0][:, -1:]).abs() < 1e-5).sum(dim=1).float().mean()),
            }
            ent.pop("full_query_top1000", None)
            del idx, gpu_s, gpu_i, runs
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            ent["error"] = f"{type(exc).__name__}: {exc}"
        out["scales"][str(n_docs)] = ent
        print(json.dumps({str(n_docs): {k2: v for k2, v in ent.items()
                                        if k2 != "exactness_sweep"}}), flush=True)
    return out


def dense_reference_topk(csr, query_dense, k, chunk=200_000):
    """Exhaustive fp32 dense reference. Scores every document, no pruning, chunked over
    documents so the dense block fits. Returns (n_queries, k) document row indices."""
    Q = torch.from_numpy(np.ascontiguousarray(query_dense)).float().to(DEVICE)
    nq = Q.shape[0]
    best_s = torch.full((nq, 0), 0.0, device=DEVICE)
    best_i = torch.zeros((nq, 0), dtype=torch.long, device=DEVICE)
    for start in range(0, csr.shape[0], chunk):
        block = csr[start:start + chunk]
        Db = torch.from_numpy(block.toarray()).float().to(DEVICE)
        s = Q @ Db.t()
        kk = min(k, s.shape[1])
        vs, ix = torch.topk(s, kk, dim=1)
        best_s = torch.cat([best_s, vs], dim=1)
        best_i = torch.cat([best_i, ix + start], dim=1)
        vs2, ix2 = torch.topk(best_s, min(k, best_s.shape[1]), dim=1)
        best_s = vs2
        best_i = torch.gather(best_i, 1, ix2)
        del Db, s
        torch.cuda.empty_cache()
    return best_i.cpu().numpy()


def run_invariant(scales):
    """The builder must leave posting lists ascending in document id, because the
    chunked path binary-searches them. The paper says this was documented but not
    enforced, and that 96.2% of lists were non-ascending before the fix."""
    from run_correctness_verification import build_gpu_index_from_csr
    out = {"_why": "verify the ascending-document-id invariant is enforced, not assumed.",
           "scales": {}}
    for n_docs in scales:
        ent = {}
        try:
            csr, _ = load_scale(n_docs)
            idx = build_gpu_index_from_csr(csr, DEVICE)
            doc_ids = idx["doc_ids"] if isinstance(idx, dict) else idx.doc_ids
            offsets = idx["offsets"] if isinstance(idx, dict) else idx.offsets
            lengths = idx["lengths"] if isinstance(idx, dict) else idx.lengths
            doc_ids = doc_ids.cpu().numpy()
            offsets = offsets.cpu().numpy()
            lengths = lengths.cpu().numpy()
            bad = 0
            nonempty = 0
            for t in range(len(lengths)):
                L = int(lengths[t])
                if L <= 1:
                    continue
                nonempty += 1
                seg = doc_ids[int(offsets[t]):int(offsets[t]) + L]
                if np.any(np.diff(seg) < 0):
                    bad += 1
            ent = {"terms_with_2plus_postings": nonempty,
                   "non_ascending_lists": bad,
                   "pct_non_ascending": 100.0 * bad / max(nonempty, 1),
                   "invariant_holds": bad == 0}
            del idx
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            ent["error"] = f"{type(exc).__name__}: {exc}"
        out["scales"][str(n_docs)] = ent
        print(json.dumps({str(n_docs): ent}), flush=True)
    return out


def run_counter_workload(which, n_docs=100_000, B=500):
    """Single-kernel workload for ncu.

    The paper reports the doc-parallel kernel at "48.9 GB per 500-query batch, measured".
    The stored counter was taken at B=50 and multiplied by ten. This runs it at B=500,
    the batch size the paper actually reports, so the figure is a measurement.

    The doc-parallel path needs the collection densified ([num_docs, 30522] fp32, about
    12.2 GiB at 100K documents), which is itself a scalability limit worth stating.
    """
    csr, meta = load_scale(n_docs)
    qi, qs, qd, _ = queries_from_meta(meta, max_terms=None)
    qi, qs = qi[:B], qs[:B]

    if which == "scatter":
        from run_correctness_verification import build_gpu_index_from_csr
        from triton_kernel import triton_fused_score
        idx = build_gpu_index_from_csr(csr, DEVICE)
        for _ in range(3):
            triton_fused_score(idx, qi, qs, top_k=1000)
    elif which == "docparallel":
        from triton_kernel_v4 import build_doc_csr_index, triton_doc_csr_score
        dense = csr.toarray().astype(np.float32)
        print(f"densified collection: {dense.nbytes / 2**30:.2f} GiB", flush=True)
        didx = build_doc_csr_index(dense, DEVICE)
        del dense
        # the doc-parallel kernel consumes DENSE query weights [B, vocab]
        qw = torch.from_numpy(np.ascontiguousarray(qd[:B])).float().to(DEVICE)
        for _ in range(3):
            triton_doc_csr_score(didx, qw, top_k=1000)
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "quality", "invariant", "counter"])
    ap.add_argument("--counter-workload", default="scatter",
                    choices=["scatter", "docparallel"])
    ap.add_argument("--scales", default="100000,500000,1000000")
    ap.add_argument("--out", default=f"{REPO}/final_results/remeasure_2026_09/results.json")
    args = ap.parse_args()

    if args.only == "counter":
        run_counter_workload(args.counter_workload)
        return

    scales = [int(x) for x in args.scales.split(",") if x]
    torch.manual_seed(0)
    res = {
        "_description": "September 2026 audit re-measurement of GPUSparse.",
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "cuda": torch.version.cuda,
        },
        "conventions": {
            "latency": "CUDA events, median of 15 after 5 warmups",
            "counters": "ncu only, collected separately",
            "reference": "exhaustive fp32 dense scoring of every document, no pruning",
        },
    }
    print(json.dumps(res["environment"]), flush=True)
    if args.only in ("all", "quality"):
        res["quality_and_exactness"] = run_quality_and_exactness(scales)
    if args.only in ("all", "invariant"):
        res["invariant"] = run_invariant(scales)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
