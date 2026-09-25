"""Does the 256-token encoder input limit explain the TREC-COVID and DBPedia deficits?

Seven of our eight small BEIR datasets land within 0.003 nDCG@10 of the Pyserini
reproduction of the same checkpoint, but TREC-COVID is 0.023 low and DBPedia 0.020 low.
Both have longer documents than the rest, and our encoder truncates document input to 256
tokens, so the natural hypothesis is that truncation is discarding content the reference
keeps. Query truncation is ruled out: neither dataset has a query above the tensor width.

This re-encodes TREC-COVID at 512 tokens and re-evaluates, holding everything else fixed.
If the gap closes, the explanation is confirmed and carries to DBPedia; if it does not, the
explanation is wrong and should not appear in the paper.
"""
import gc, json, os, sys, time
import numpy as np, scipy.sparse as sp, torch

W = os.environ.get("GPUSPARSE_DATA_BEIR13", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/beir13"))
GS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{GS}/src"); sys.path.insert(0, GS)
sys.path.insert(0, W)
OUT = f"{W}/results_doclen.json"
DEV = "cuda:0"

import ir_datasets
from run_all_todo_experiments import encode_splade_to_csr
from triton_kernel import triton_fused_score
from run_correctness_verification import build_gpu_index_from_csr

DATASET = "beir/trec-covid"
REFERENCE = 0.727


def ndcg_at(run_ids, qrels, qids, k=10):
    tot, n = 0.0, 0
    for i, qid in enumerate(qids):
        rel = qrels.get(str(qid), {})
        if not rel:
            continue
        gains = [rel.get(str(d), 0) for d in run_ids[i][:k]]
        dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        if idcg > 0:
            v = dcg / idcg
            tot += min(v, 1.0); n += 1
    return (tot / n if n else 0.0), n


ds = ir_datasets.load(DATASET)
docs, dids = [], []
for d in ds.docs_iter():
    ti = getattr(d, "title", "") or ""
    tx = getattr(d, "text", "") or ""
    docs.append((ti + " " + tx).strip()); dids.append(d.doc_id)
seen, keep = set(), []
for i, x in enumerate(dids):
    if x not in seen:
        seen.add(x); keep.append(i)
docs = [docs[i] for i in keep]; dids = [dids[i] for i in keep]

qt, qids = [], []
for q in ds.queries_iter():
    qt.append(getattr(q, "text", str(q))); qids.append(q.query_id)
qrels = {}
for qr in ds.qrels_iter():
    qrels.setdefault(str(qr.query_id), {})[str(qr.doc_id)] = int(qr.relevance)

# how much content does each limit actually keep?
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
lens = [len(tok(d, truncation=False)["input_ids"]) for d in docs[:4000]]
lens = np.array(lens)
print(f"{DATASET}: {len(docs):,} docs, {len(qids)} queries")
print(f"  token length on a 4,000-doc sample: mean {lens.mean():.0f}, median "
      f"{np.median(lens):.0f}, p90 {np.percentile(lens,90):.0f}, max {lens.max()}")
for lim in (256, 512):
    print(f"  fraction exceeding {lim} tokens: {(lens>lim).mean():.1%}")
print(flush=True)

res = {"dataset": DATASET, "reference_ndcg10": REFERENCE, "n_docs": len(docs),
       "token_len_mean": float(lens.mean()), "token_len_p90": float(np.percentile(lens,90)),
       "frac_over_256": float((lens>256).mean()), "frac_over_512": float((lens>512).mean()),
       "runs": {}}

for max_len in (256, 512):
    t0 = time.time()
    doc_csr = encode_splade_to_csr(docs, DEV, batch_size=256, max_length=max_len)
    q_csr = encode_splade_to_csr(qt, DEV, batch_size=256, max_length=max_len)
    enc_s = time.time() - t0

    nq = q_csr.shape[0]
    MT = int(np.diff(q_csr.tocsr().indptr).max())
    qc = q_csr.tocsr()
    ti = np.full((nq, MT), -1, dtype=np.int32); ts = np.zeros((nq, MT), dtype=np.float32)
    for i in range(nq):
        row = qc.getrow(i); o = np.argsort(-row.data)[:MT]
        ti[i, :len(o)] = row.indices[o]; ts[i, :len(o)] = row.data[o]
    qi = torch.from_numpy(ti).to(DEV); qs = torch.from_numpy(ts).to(DEV)

    idx = build_gpu_index_from_csr(doc_csr.tocsr().astype(np.float32), DEV)
    ids = []
    for s0 in range(0, nq, 200):
        _, i2 = triton_fused_score(idx, qi[s0:s0+200].contiguous(),
                                   qs[s0:s0+200].contiguous(),
                                   top_k=min(1000, doc_csr.shape[0]))
        ids.append(i2.cpu().numpy())
    run = np.concatenate(ids, 0)
    run_dids = [[dids[j] for j in row if str(dids[j]) != str(qids[r])]
                for r, row in enumerate(run)]
    nd, nev = ndcg_at(run_dids, qrels, qids, 10)

    res["runs"][f"max_length_{max_len}"] = {
        "ndcg@10": nd, "queries_evaluated": nev, "nnz": int(doc_csr.nnz),
        "terms_per_doc": float(doc_csr.nnz/doc_csr.shape[0]), "encode_s": enc_s,
        "delta_vs_reference": nd - REFERENCE}
    print(f"  max_length={max_len}: nDCG@10 = {nd:.4f}  (reference {REFERENCE}, "
          f"delta {nd-REFERENCE:+.4f}), {doc_csr.nnz/doc_csr.shape[0]:.1f} terms/doc, "
          f"encoded in {enc_s:.0f}s", flush=True)
    del idx, doc_csr, q_csr; torch.cuda.empty_cache(); gc.collect()

a = res["runs"]["max_length_256"]["ndcg@10"]; b = res["runs"]["max_length_512"]["ndcg@10"]
print(f"\n256 -> 512 tokens moves nDCG@10 by {b-a:+.4f}; the gap to the reference was "
      f"{a-REFERENCE:+.4f}")
print("HYPOTHESIS CONFIRMED" if b - a > 0.01 else "HYPOTHESIS NOT SUPPORTED")
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
