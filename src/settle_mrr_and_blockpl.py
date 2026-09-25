"""Settle two numbers the audit left open.

1. MRR@10 at 100K: 0.892 or 0.888?
   `tab:quality` prints 0.892 and `tab:comparison` prints 0.888 for what looks
   like the same configuration. Two independent project tracker notes record
   0.888 as the intended figure, and nothing in the project record mentions
   0.892, yet the September re-measurement measured 0.8916 with official qrels.

   Hypothesis under test: 0.888 came from the same truncating harness that
   produced the retracted 0.999 recall figures. The query tensors were built at
   a fixed width of 64 terms, which truncates 67 of these 500 queries (the
   longest has 107 non-zero terms). If that is the cause, MRR@10 at
   max_terms=64 should land near 0.888 and the untruncated value near 0.892,
   which would tie both discrepancies to one root cause and settle both.

   This sweeps the query width and reports MRR@10, nDCG@10 and Recall@1000 at
   each, so the answer is a measurement rather than an inference from notes.

2. The BLOCK_PL claim. The paper says values from 64 to 512 land within 15% of
   each other with 128 chosen by grid search. No artifact anywhere records a
   block_pl value or a sweep, so the claim is currently unsupported. This
   measures it. If the spread is wider than 15%, the sentence changes.

Latency is CUDA-event timed, median of 15 after 5 warmups.
"""

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
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def queries_from_meta(meta, max_terms=None):
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
        ti[i, :len(order)] = idx[order]
        ts[i, :len(order)] = vals[order]
    stats = {"n_queries": nq, "max_nnz": int(longest), "width_used": int(width),
             "n_truncated": int(sum(len(x) > width for x in nnz))}
    return torch.from_numpy(ti).to(DEVICE), torch.from_numpy(ts).to(DEVICE), stats


def mrr_at_k(ranked, qids, qrels, k=10):
    tot = 0.0
    for i, qid in enumerate(qids):
        rel = qrels.get(str(qid), qrels.get(qid, {})) or {}
        rel = {str(d) for d, g in rel.items() if float(g) > 0}
        if not rel:
            continue
        for r, did in enumerate(ranked[i][:k]):
            if str(did) in rel:
                tot += 1.0 / (r + 1)
                break
    return tot / len(qids)


def ndcg_at_k(ranked, qids, qrels, k=10):
    out = 0.0
    for i, qid in enumerate(qids):
        rel = qrels.get(str(qid), qrels.get(qid, {})) or {}
        rel = {str(d): float(g) for d, g in rel.items() if float(g) > 0}
        if not rel:
            continue
        dcg = sum(rel.get(str(d), 0.0) / np.log2(r + 2)
                  for r, d in enumerate(ranked[i][:k]))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        if idcg > 0:
            out += dcg / idcg
    return out / len(qids)


def recall_at_k(ranked, qids, qrels, k=1000):
    out = 0.0
    for i, qid in enumerate(qids):
        rel = qrels.get(str(qid), qrels.get(qid, {})) or {}
        rel = {str(d) for d, g in rel.items() if float(g) > 0}
        if not rel:
            continue
        out += len(rel & {str(d) for d in ranked[i][:k]}) / len(rel)
    return out / len(qids)


def main():
    from triton_kernel import triton_fused_score
    from run_correctness_verification import build_gpu_index_from_csr

    n_docs = 100_000
    csr = sp.load_npz(f"{CACHE}/msmarco_splade_csr_{n_docs}.npz").tocsr().astype(np.float32)
    meta = torch.load(f"{CACHE}/msmarco_splade_csr_{n_docs}_meta.pt",
                      map_location="cpu", weights_only=False)
    qids, qrels = meta["qids"], meta["qrels"]
    idx = build_gpu_index_from_csr(csr, DEVICE)
    k = 1000

    res = {"_why": __doc__.split("\n\n")[0],
           "environment": {"gpu": torch.cuda.get_device_name(0),
                           "torch": torch.__version__,
                           "triton": __import__("triton").__version__},
           "n_docs": n_docs, "quality_vs_query_width": {}, "block_pl_sweep": {}}

    print("########## 1: is 0.888 the truncated reading of 0.892? ##########", flush=True)
    for mt in (64, 96, 107, 128, None):
        qi, qs, st = queries_from_meta(meta, max_terms=mt)
        _, gi = triton_fused_score(idx, qi, qs, top_k=k)
        ranked = gi.cpu().numpy().tolist()
        ent = {"query_stats": st,
               "mrr@10": mrr_at_k(ranked, qids, qrels, 10),
               "ndcg@10": ndcg_at_k(ranked, qids, qrels, 10),
               "recall@1000": recall_at_k(ranked, qids, qrels, k)}
        res["quality_vs_query_width"][str(mt)] = ent
        print(f"  max_terms={str(mt):>4} truncated={st['n_truncated']:>3}  "
              f"MRR@10={ent['mrr@10']:.4f}  nDCG@10={ent['ndcg@10']:.4f}  "
              f"R@1000={ent['recall@1000']:.4f}", flush=True)
        del gi
        torch.cuda.empty_cache()

    print("\n########## 2: the BLOCK_PL claim ##########", flush=True)
    qi, qs, _ = queries_from_meta(meta, max_terms=None)
    base = None
    for bpl in (32, 64, 128, 256, 512, 1024):
        try:
            ms = t_ms(lambda: triton_fused_score(idx, qi, qs, top_k=k, block_pl=bpl))
            _, gi = triton_fused_score(idx, qi, qs, top_k=k, block_pl=bpl)
            # every block size must return the same ranking
            if base is None:
                base = gi.cpu().numpy()
                agree = 1.0
            else:
                g = gi.cpu().numpy()
                agree = float(np.mean([
                    len(set(base[r, :10].tolist()) & set(g[r, :10].tolist())) / 10
                    for r in range(base.shape[0])]))
            res["block_pl_sweep"][str(bpl)] = {"latency_ms": ms, "top10_agreement": agree}
            print(f"  block_pl={bpl:>5}  {ms:>8.4f} ms  top10_agree={agree:.4f}", flush=True)
            del gi
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            res["block_pl_sweep"][str(bpl)] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"  block_pl={bpl:>5}  FAILED: {type(exc).__name__}", flush=True)

    ok = {int(b): v["latency_ms"] for b, v in res["block_pl_sweep"].items()
          if "latency_ms" in v}
    if ok:
        lo, hi = min(ok.values()), max(ok.values())
        best = min(ok, key=ok.get)
        res["block_pl_summary"] = {
            "fastest_block_pl": best, "min_ms": lo, "max_ms": hi,
            "spread_pct_over_all": 100.0 * (hi - lo) / lo,
            "spread_pct_64_to_512": (
                100.0 * (max(ok[b] for b in (64, 128, 256, 512) if b in ok)
                         - min(ok[b] for b in (64, 128, 256, 512) if b in ok))
                / min(ok[b] for b in (64, 128, 256, 512) if b in ok)
                if all(b in ok for b in (64, 128, 256, 512)) else None),
        }
        print(f"\n  fastest={best}  spread over all sizes={res['block_pl_summary']['spread_pct_over_all']:.1f}%"
              f"  spread 64..512={res['block_pl_summary']['spread_pct_64_to_512']}", flush=True)

    out = f"{REPO}/final_results/remeasure_2026_09/mrr_and_blockpl.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
