"""Full-corpus BEIR evaluation for GPUSparse. No truncation, incremental checkpointing.

The published table used 3 datasets and truncated TREC-COVID to the first 50,000 of
171,332 documents while silently discarding qrels for dropped documents, which is why
its nDCG@10 of 0.297 sits far below published SPLADE++ results. This runs the standard
13-dataset BEIR subset on full corpora.
"""
import json, os, sys, time, gc
import numpy as np, scipy.sparse as sp, torch

GS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{GS}/src"); sys.path.insert(0, GS)
W = os.environ.get("GPUSPARSE_DATA_BEIR13", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/beir13"))
CACHE = f"{W}/cache_v2"; OUT = f"{W}/results_v2.json"
DEV = "cuda:0"
GROUP = sys.argv[1] if len(sys.argv) > 1 else "small"

SMALL = [("beir/nfcorpus/test",3633),("beir/scifact/test",5183),("beir/arguana",8674),
         ("beir/scidocs",25657),("beir/fiqa/test",57638),("beir/trec-covid",171332),
         ("beir/webis-touche2020/v2",382545),("beir/quora/test",522931)]
LARGE = [("beir/nq",2681468),("beir/dbpedia-entity/test",4635922),
         ("beir/hotpotqa/test",5233329),("beir/fever/test",5416568),
         ("beir/climate-fever",5416593)]
# "one:<name>" runs a single dataset into its own results file. The five large
# corpora take ~2.5 h of SPLADE encoding each, so they run as parallel one-GPU jobs;
# a shared results file would have them clobbering each other's writes.
if GROUP.startswith("one:"):
    want = GROUP[4:]
    DATASETS = [d for d in (SMALL + LARGE) if d[0] == want]
    if not DATASETS:
        sys.exit(f"unknown dataset {want!r}; choose from "
                 + ", ".join(d[0] for d in SMALL + LARGE))
    OUT = f"{W}/results_v2__{want.replace('/','_')}.json"
else:
    DATASETS = SMALL if GROUP == "small" else LARGE

from run_all_todo_experiments import encode_splade_to_csr
from triton_kernel import triton_fused_score
from run_correctness_verification import build_gpu_index_from_csr

def ndcg_at(run_ids, run_scores, qrels, qids, k=10):
    import math
    tot, n = 0.0, 0
    for i, qid in enumerate(qids):
        rel = qrels.get(str(qid), {})
        if not rel: continue
        n += 1
        seen, uniq = set(), []
        for d in run_ids[i]:
            if d not in seen:
                seen.add(d); uniq.append(d)
            if len(uniq) >= k: break
        # Linear gain, matching trec_eval's ndcg_cut, which is what Pyserini and the
        # official BEIR evaluator report. trec_eval's m_ndcg_cut.c uses
        # `gain = results_rel_list[i]` and states "Gain values are the relevance values
        # in the qrels file", with discount log2(rank+1) and an ideal formed by sorting
        # the judged relevance values descending and truncating at the cutoff.
        #
        # This previously used the exponential gain 2**g - 1. That is identical for
        # binary qrels, so it was invisible on most of BEIR, but it understates nDCG on
        # the graded collections and accounted for the entire apparent deficit on
        # TREC-COVID (0.7043 exponential against 0.7282 linear, reference 0.727) and
        # most of DBPedia's.
        gains = [rel.get(str(d), 0) for d in uniq]
        dcg = sum(g/math.log2(r+2) for r, g in enumerate(gains) if g > 0)
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g/math.log2(r+2) for r, g in enumerate(ideal) if g > 0)
        v = (dcg/idcg) if idcg > 0 else 0.0
        assert v <= 1.0 + 1e-6, f"nDCG {v} > 1 for {qid}: duplicate doc ids in the ranking?"
        tot += v
    return tot/max(n,1), n

def recall_at(run_ids, qrels, qids, k=1000):
    tot, n = 0.0, 0
    for i, qid in enumerate(qids):
        rel = {d for d,g in qrels.get(str(qid), {}).items() if g > 0}
        if not rel: continue
        n += 1
        tot += len(rel & {str(d) for d in run_ids[i][:k]})/len(rel)
    return tot/max(n,1)

# FORCE_EVAL=1 re-scores and re-evaluates every dataset from the cached encodings,
# which is what to use after changing evaluation logic (e.g. adding self-match
# exclusion) without paying to re-encode the corpora.
FORCE_EVAL = os.environ.get("FORCE_EVAL") == "1"
res = {} if FORCE_EVAL else (json.load(open(OUT)) if os.path.exists(OUT) else {})
import ir_datasets
for name, approx_n in DATASETS:
    key = name.replace("/","_")
    if key in res:
        print(f"[skip] {name} already done", flush=True); continue
    print(f"\n{'='*70}\n{name}  (~{approx_n:,} docs, FULL corpus)\n{'='*70}", flush=True)
    t_all = time.time()
    csr_p, meta_p = f"{CACHE}/{key}_docs.npz", f"{CACHE}/{key}_meta.pt"
    try:
        if os.path.exists(csr_p) and os.path.exists(meta_p):
            doc_csr = sp.load_npz(csr_p); meta = torch.load(meta_p, map_location="cpu", weights_only=False)
            print(f"  cached: {doc_csr.shape[0]:,} docs", flush=True)
        else:
            ds = ir_datasets.load(name)
            texts, dids = [], []
            for d in ds.docs_iter():
                t = getattr(d,"text",None) or getattr(d,"body",None) or ""
                ti = getattr(d,"title","") or ""
                # No character truncation: 512 chars is about 100 tokens, well under
                # the encoder's 256-token window, and it cost 5-17% relative nDCG.
                # Let the tokenizer truncate at the token level instead.
                texts.append((ti + " " + t).strip()); dids.append(d.doc_id)
            # Duplicate doc ids are either a corrupt local store (concurrent writers)
            # or, for a few BEIR variants, genuine upstream duplication. Deduplicate
            # rather than abort, but report it loudly: silently scoring duplicates
            # inflates DCG above the ideal and yields nDCG > 1.
            if len(set(dids)) != len(dids):
                n_before = len(dids)
                seen, keep = set(), []
                for i, d in enumerate(dids):
                    if d not in seen:
                        seen.add(d); keep.append(i)
                dids = [dids[i] for i in keep]
                texts = [texts[i] for i in keep]
                print(f"  WARNING: doc store yielded {n_before:,} docs, {len(dids):,} unique. "
                      f"Deduplicated {n_before - len(dids):,}. If the ratio is a clean integer "
                      f"multiple, IR_DATASETS_HOME was populated by concurrent processes and "
                      f"should be deleted and refetched single-process.", flush=True)
            print(f"  loaded {len(texts):,} docs; encoding SPLADE...", flush=True)
            t0=time.time(); doc_csr = encode_splade_to_csr(texts, DEV, batch_size=256)
            print(f"  encoded in {time.time()-t0:.0f}s ({len(texts)/(time.time()-t0):.0f} docs/s)", flush=True)
            qt, qids = [], []
            for q in ds.queries_iter():
                qt.append(getattr(q,"text",str(q))); qids.append(q.query_id)
            q_csr = encode_splade_to_csr(qt, DEV, batch_size=256)
            qrels = {}
            for qr in ds.qrels_iter():
                qrels.setdefault(str(qr.query_id), {})[str(qr.doc_id)] = int(qr.relevance)
            meta = {"query_csr": q_csr, "qids": qids, "qrels": qrels, "doc_ids": dids}
            sp.save_npz(csr_p, doc_csr); torch.save(meta, meta_p)
            del texts, qt
        gc.collect()

        dids = meta["doc_ids"]; qids = meta["qids"]; qrels = meta["qrels"]
        q_csr = meta["query_csr"]
        # Query tensors sized to the longest query actually present, so no query term is
        # dropped. A fixed cap of 128 was previously used with a comment asserting it did
        # not truncate real queries; that holds on MS MARCO (max 107 terms) but not on
        # BEIR. ArguAna queries are whole arguments and average 216.7 SPLADE terms with a
        # maximum of 326, so 99.5% of them were being cut, and SciFact, SCIDOCS and Quora
        # lost a few as well. Keeping the top-weighted terms makes the cut nearly harmless
        # in practice, but it is still an approximation and this harness claims exactness.
        nq = q_csr.shape[0]
        MT = int(np.diff(q_csr.indptr).max()) if nq else 1
        n_cut = int((np.diff(q_csr.indptr) > MT).sum())
        assert n_cut == 0, f"{n_cut} queries exceed MT={MT}"
        print(f"  query terms: mean {np.diff(q_csr.indptr).mean():.1f}, max {MT} "
              f"(untruncated)", flush=True)
        ti = np.full((nq,MT),-1,dtype=np.int32); ts = np.zeros((nq,MT),dtype=np.float32)
        for i in range(nq):
            row = q_csr.getrow(i); idx, val = row.indices, row.data
            o = np.argsort(-val)[:MT]; ti[i,:len(o)] = idx[o]; ts[i,:len(o)] = val[o]
        qi = torch.from_numpy(ti).to(DEV); qs = torch.from_numpy(ts).to(DEV)

        idx = build_gpu_index_from_csr(doc_csr.tocsr().astype(np.float32), DEV)
        B = 200
        all_ids = []
        # Warm up before timing: the first triton_fused_score call in a process pays
        # Triton JIT compilation, which showed up as 1.599 ms/q for whichever dataset
        # ran first against 0.003 ms/q for the next one of similar size.
        _ = triton_fused_score(idx, qi[:min(B, nq)].contiguous(),
                               qs[:min(B, nq)].contiguous(),
                               top_k=min(1000, doc_csr.shape[0]))
        torch.cuda.synchronize()
        t0 = time.time()
        for s0 in range(0, nq, B):
            _, ids = triton_fused_score(idx, qi[s0:s0+B].contiguous(), qs[s0:s0+B].contiguous(),
                                        top_k=min(1000, doc_csr.shape[0]))
            all_ids.append(ids.cpu().numpy())
        torch.cuda.synchronize(); el = time.time()-t0
        run = np.concatenate(all_ids, 0)
        # Drop the query's own document from its ranking, which is what the official
        # BEIR evaluator does for every dataset. It matters most on ArguAna, where
        # 92.3% of query ids also exist as doc ids and none is its own relevance
        # judgment: the self-match takes rank 1 and pushes the true counter-argument
        # down, costing about 0.09 nDCG@10.
        n_self = 0
        run_dids = []
        for r, row in enumerate(run):
            qid = str(qids[r])
            keep = []
            for j in row:
                dj = dids[j]
                if str(dj) == qid:
                    n_self += 1
                    continue
                keep.append(dj)
            run_dids.append(keep)
        if n_self:
            print(f"  removed {n_self} self-matches "
                  f"({n_self/max(nq,1):.2f} per query)", flush=True)
        nd, nev = ndcg_at(run_dids, None, qrels, qids, 10)
        rc = recall_at(run_dids, qrels, qids, 1000)
        res[key] = {"dataset":name,"num_docs":int(doc_csr.shape[0]),"num_queries":nq,
                    "num_evaluated":nev,"ndcg@10":nd,"recall@1000":rc,
                    "ms_per_query":el/nq*1000,"qps":nq/el,
                    "total_s":time.time()-t_all,"truncated":False,"self_matches_removed":int(n_self),"query_terms_max":int(MT),"query_truncated":False}
        print(f"  RESULT {name}: docs={doc_csr.shape[0]:,} q={nq} nDCG@10={nd:.4f} "
              f"R@1000={rc:.4f} {el/nq*1000:.3f} ms/q", flush=True)
        del idx, doc_csr; torch.cuda.empty_cache(); gc.collect()
    except Exception as ex:
        import traceback; traceback.print_exc()
        res[key] = {"dataset":name,"error":f"{type(ex).__name__}: {ex}"}
        print(f"  FAILED {name}: {ex}", flush=True)
        torch.cuda.empty_cache(); gc.collect()
    json.dump(res, open(OUT,"w"), indent=2)
print(f"\nwrote {OUT}", flush=True)
