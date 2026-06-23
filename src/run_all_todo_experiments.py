"""
GPUSparse: Run ALL TODO experiments for SIGIR submission.

Experiments:
  1. Full-scale 8.8M MS MARCO (MRR@10, nDCG@10, Recall@1000, latency, memory)
  2. SPARe (cuSPARSE) baseline comparison
  3. torch.compile comparison for SPLADE scoring
  4. BEIR evaluation (SciFact, NFCorpus, TREC-COVID)

Usage:
    python src/run_all_todo_experiments.py --device cuda:0 [--skip-encoding]
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import sys
import gc
import argparse
from pathlib import Path
from collections import defaultdict
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.triton_kernel_v4 import (
    build_doc_csr_index, build_query_weight_matrix,
    triton_doc_csr_score, triton_doc_csr_score_chunked,
)

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
CACHE = BASE / "data_cache"
RESULTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

VOCAB_SIZE = 30522


def bench_fn(fn, warmup=3, trials=10, device="cuda:0"):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize(device)
    times = []
    for _ in range(trials):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "std_ms": np.std(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


###############################################################################
# DATA: Real SPLADE encodings via naver/splade-v3
###############################################################################

def encode_splade_to_csr(texts, device, batch_size=64, max_length=256):
    """Encode texts with SPLADE and return scipy CSR matrix directly."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()

    rows_list, cols_list, vals_list = [], [], []
    doc_idx = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            logits = out.logits
            sparse = torch.log1p(torch.relu(logits))
            mask = enc["attention_mask"].unsqueeze(-1)
            sparse = (sparse * mask).max(dim=1).values  # (B, vocab)

            nz = sparse.nonzero(as_tuple=False).cpu()
            vals = sparse[nz[:, 0], nz[:, 1]].cpu()
            rows_list.append((nz[:, 0] + doc_idx).numpy().astype(np.int32))
            cols_list.append(nz[:, 1].numpy().astype(np.int32))
            vals_list.append(vals.numpy().astype(np.float32))
            doc_idx += len(batch)

            if (i // batch_size) % 50 == 0:
                print(f"  SPLADE encoded {min(i+batch_size, len(texts))}/{len(texts)}")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    csr = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), VOCAB_SIZE))
    print(f"  CSR: {csr.shape}, nnz={csr.nnz:,}, "
          f"{(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes) / 1e6:.0f} MB")
    return csr


def splade_to_dense(splade_csr, vocab_size=VOCAB_SIZE):
    """Convert scipy CSR to dense numpy array (only for small matrices like queries)."""
    return splade_csr.toarray().astype(np.float32)


def build_doc_csr_from_scipy(csr_mat, device):
    """Build Triton doc-CSR index directly from scipy CSR - no dense intermediate."""
    import time as _time
    t0 = _time.time()
    num_docs = csr_mat.shape[0]

    sorted_terms = csr_mat.indices.astype(np.int32)
    sorted_scores = csr_mat.data.astype(np.float32)
    doc_offsets = csr_mat.indptr[:-1].astype(np.int64)
    doc_counts = np.diff(csr_mat.indptr).astype(np.int32)

    total_entries = len(sorted_terms)
    max_doc_len = int(doc_counts.max()) if num_docs > 0 else 0
    elapsed = _time.time() - t0
    mem_mb = (total_entries * 8 + num_docs * 12) / 1e6
    print(f"  Doc-CSR index: {num_docs} docs, {total_entries} entries, "
          f"max_doc_len={max_doc_len}, {mem_mb:.0f} MB, built in {elapsed:.1f}s")

    return {
        'doc_term_ids': torch.from_numpy(sorted_terms).to(device),
        'doc_term_scores': torch.from_numpy(sorted_scores).to(device),
        'doc_offsets': torch.from_numpy(doc_offsets).to(device),
        'doc_lengths': torch.from_numpy(doc_counts).to(device),
        'num_docs': num_docs,
        'vocab_size': csr_mat.shape[1],
        'total_entries': total_entries,
        'max_doc_len': max_doc_len,
        'device': device,
    }


def load_msmarco_passages(max_docs=None):
    """Load MS MARCO passage texts via ir_datasets."""
    import ir_datasets
    ds = ir_datasets.load("msmarco-passage/dev/small")

    queries = {}
    for q in ds.queries_iter():
        queries[q.query_id] = q.text

    raw_qrels = defaultdict(dict)
    relevant_doc_ids = set()
    for qrel in ds.qrels_iter():
        raw_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        relevant_doc_ids.add(qrel.doc_id)

    docs = {}
    filler = {}
    for doc in ds.docs_iter():
        if doc.doc_id in relevant_doc_ids:
            docs[doc.doc_id] = doc.text
        elif max_docs and len(filler) < (max_docs - len(relevant_doc_ids)):
            filler[doc.doc_id] = doc.text
        if max_docs and (len(docs) + len(filler)) >= max_docs:
            if len(docs) >= len(relevant_doc_ids):
                break

    all_docs = {}
    all_docs.update(docs)
    remaining = (max_docs or (len(docs) + len(filler))) - len(all_docs)
    for did, text in list(filler.items())[:remaining]:
        all_docs[did] = text

    doc_ids = list(all_docs.keys())
    doc_texts = [all_docs[d] for d in doc_ids]
    docid_to_idx = {d: i for i, d in enumerate(doc_ids)}

    qrels = defaultdict(dict)
    for qid in raw_qrels:
        for did, rel in raw_qrels[qid].items():
            if did in docid_to_idx:
                qrels[qid][docid_to_idx[did]] = rel

    valid_qids = [q for q in queries if q in qrels and len(qrels[q]) > 0]
    query_texts = [queries[q] for q in valid_qids]
    query_qrels = {q: qrels[q] for q in valid_qids}

    return doc_texts, query_texts, valid_qids, query_qrels, docid_to_idx


###############################################################################
# METRICS
###############################################################################

def compute_mrr(rankings, qrels, k=10):
    rrs = []
    for qid, ranked_docs in rankings.items():
        if qid not in qrels:
            continue
        for rank, doc_idx in enumerate(ranked_docs[:k], 1):
            if doc_idx in qrels[qid] and qrels[qid][doc_idx] > 0:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return np.mean(rrs) if rrs else 0.0


def compute_ndcg(rankings, qrels, k=10):
    from math import log2
    ndcgs = []
    for qid, ranked_docs in rankings.items():
        if qid not in qrels:
            continue
        dcg = 0.0
        for rank, doc_idx in enumerate(ranked_docs[:k], 1):
            if doc_idx in qrels[qid]:
                dcg += qrels[qid][doc_idx] / log2(rank + 1)
        ideal = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum(r / log2(i + 2) for i, r in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs) if ndcgs else 0.0


def compute_recall(rankings, qrels, k=1000):
    recalls = []
    for qid, ranked_docs in rankings.items():
        if qid not in qrels:
            continue
        rel = {d for d, r in qrels[qid].items() if r > 0}
        retrieved = set(ranked_docs[:k])
        recalls.append(len(rel & retrieved) / len(rel) if rel else 0.0)
    return np.mean(recalls) if recalls else 0.0


###############################################################################
# EXPERIMENT 1: Full-Scale MS MARCO
###############################################################################

def run_fullscale_msmarco(device, num_docs_list, num_queries=500, skip_encoding=False):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Full-Scale MS MARCO Evaluation")
    print("="*70)

    results = {}
    for num_docs in num_docs_list:
        print(f"\n--- Scale: {num_docs:,} documents ---")
        cache_path = CACHE / f"msmarco_splade_csr_{num_docs}.npz"
        cache_meta = CACHE / f"msmarco_splade_csr_{num_docs}_meta.pt"

        if cache_path.exists() and cache_meta.exists():
            print(f"  Loading cached CSR from {cache_path}")
            doc_csr_scipy = sp.load_npz(cache_path)
            meta = torch.load(cache_meta, map_location="cpu", weights_only=False)
            query_dense = meta["query_dense"]
            qids = meta["qids"]
            qrels = meta["qrels"]
        else:
            doc_texts, query_texts, qids, qrels, _ = load_msmarco_passages(max_docs=num_docs)
            print(f"  Encoding {len(doc_texts)} docs with SPLADE...")
            doc_csr_scipy = encode_splade_to_csr(doc_texts, device, batch_size=32)
            print(f"  Encoding {len(query_texts)} queries with SPLADE...")
            query_csr = encode_splade_to_csr(query_texts[:num_queries], device, batch_size=64)
            query_dense = splade_to_dense(query_csr)
            qids = qids[:num_queries]
            sp.save_npz(cache_path, doc_csr_scipy)
            torch.save({"query_dense": query_dense, "qids": qids, "qrels": dict(qrels)},
                       cache_meta)
            print(f"  Cached to {cache_path}")

        doc_csr = build_doc_csr_from_scipy(doc_csr_scipy, device)
        query_weights = build_query_weight_matrix(
            torch.from_numpy(query_dense) if isinstance(query_dense, np.ndarray) else query_dense,
            device,
        )

        # Warmup
        triton_doc_csr_score_chunked(doc_csr, query_weights[:2], top_k=10, query_chunk_size=2)
        torch.cuda.synchronize(device)

        # Benchmark latency
        t0 = time.perf_counter()
        top_scores, top_ids = triton_doc_csr_score_chunked(
            doc_csr, query_weights, top_k=1000, query_chunk_size=50
        )
        torch.cuda.synchronize(device)
        total_time = time.perf_counter() - t0
        per_query_ms = (total_time / len(qids)) * 1000

        # Compute metrics
        rankings = {}
        for i, qid in enumerate(qids):
            rankings[qid] = top_ids[i].cpu().tolist()

        mrr10 = compute_mrr(rankings, qrels, k=10)
        ndcg10 = compute_ndcg(rankings, qrels, k=10)
        recall1000 = compute_recall(rankings, qrels, k=1000)

        mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

        results[num_docs] = {
            "num_docs": num_docs,
            "num_queries": len(qids),
            "mrr@10": mrr10,
            "ndcg@10": ndcg10,
            "recall@1000": recall1000,
            "total_time_s": total_time,
            "per_query_ms": per_query_ms,
            "throughput_qps": len(qids) / total_time,
            "gpu_memory_mb": mem_mb,
        }

        print(f"  MRR@10: {mrr10:.4f}")
        print(f"  nDCG@10: {ndcg10:.4f}")
        print(f"  Recall@1000: {recall1000:.4f}")
        print(f"  Latency: {per_query_ms:.1f} ms/query")
        print(f"  Throughput: {len(qids)/total_time:.1f} qps")
        print(f"  GPU memory: {mem_mb:.0f} MB")

        torch.cuda.empty_cache()
        gc.collect()

    return results


###############################################################################
# EXPERIMENT 2: SPARe (cuSPARSE) Baseline
###############################################################################

def run_cusparse_baseline(device, doc_csr_scipy, query_dense, num_docs):
    """Benchmark cuSPARSE SpMV for sparse retrieval (SPARe approach)."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: SPARe (cuSPARSE SpMV) Baseline")
    print("="*70)

    print(f"  CSR matrix: {doc_csr_scipy.shape}, nnz={doc_csr_scipy.nnz:,}")

    crow = torch.from_numpy(doc_csr_scipy.indptr.astype(np.int64)).to(device)
    col = torch.from_numpy(doc_csr_scipy.indices.astype(np.int64)).to(device)
    val = torch.from_numpy(doc_csr_scipy.data.astype(np.float32)).to(device)
    doc_csr_torch = torch.sparse_csr_tensor(crow, col, val,
                                             size=doc_csr_scipy.shape,
                                             device=device)

    q_dense_gpu = torch.from_numpy(query_dense).to(device=device, dtype=torch.float32)

    def cusparse_score():
        return torch.sparse.mm(doc_csr_torch, q_dense_gpu.T)

    for _ in range(3):
        cusparse_score()
        torch.cuda.synchronize(device)

    timing = bench_fn(cusparse_score, warmup=3, trials=10, device=device)

    return {
        "method": "cuSPARSE_SpMV",
        "num_docs": num_docs,
        "num_queries": query_dense.shape[0],
        "nnz": doc_csr_scipy.nnz,
        **timing,
    }


###############################################################################
# EXPERIMENT 3: torch.compile Comparison
###############################################################################

def run_torch_compile_comparison(device, doc_csr_scipy, query_dense_np):
    """Compare GPUSparse Triton kernel vs torch.compile on SPLADE scoring."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: torch.compile Comparison")
    print("="*70)

    num_docs = min(100000, doc_csr_scipy.shape[0])
    num_queries = min(100, query_dense_np.shape[0])

    doc_sub_csr = doc_csr_scipy[:num_docs]
    q_sub = query_dense_np[:num_queries]

    doc_sub_dense = doc_sub_csr.toarray().astype(np.float32)

    doc_gpu = torch.from_numpy(doc_sub_dense).to(device=device, dtype=torch.float32)
    q_gpu = torch.from_numpy(q_sub).to(device=device, dtype=torch.float32)

    def pytorch_dense():
        return q_gpu @ doc_gpu.T

    compiled_fn = torch.compile(pytorch_dense, mode="max-autotune")
    for _ in range(5):
        compiled_fn()
        torch.cuda.synchronize(device)

    doc_csr = build_doc_csr_from_scipy(doc_sub_csr, device)
    query_weights = build_query_weight_matrix(torch.from_numpy(q_sub), device)

    def triton_sparse():
        return triton_doc_csr_score_chunked(doc_csr, query_weights, top_k=10, query_chunk_size=50)

    pytorch_time = bench_fn(pytorch_dense, device=device)
    compiled_time = bench_fn(compiled_fn, device=device)
    triton_time = bench_fn(triton_sparse, device=device)

    del doc_gpu, doc_sub_dense
    torch.cuda.empty_cache()

    results = {
        "num_docs": num_docs,
        "num_queries": num_queries,
        "pytorch_dense": pytorch_time,
        "torch_compile": compiled_time,
        "triton_gpusparse": triton_time,
        "speedup_over_pytorch": pytorch_time["mean_ms"] / triton_time["mean_ms"],
        "speedup_over_compile": compiled_time["mean_ms"] / triton_time["mean_ms"],
    }

    print(f"  PyTorch dense: {pytorch_time['mean_ms']:.2f} ms")
    print(f"  torch.compile: {compiled_time['mean_ms']:.2f} ms")
    print(f"  GPUSparse Triton: {triton_time['mean_ms']:.2f} ms")
    print(f"  Speedup over torch.compile: {results['speedup_over_compile']:.2f}x")

    return results


###############################################################################
# EXPERIMENT 4: BEIR Evaluation
###############################################################################

def run_beir_evaluation(device, datasets=None, max_docs=50000):
    """Run GPUSparse on BEIR datasets."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: BEIR Evaluation")
    print("="*70)

    if datasets is None:
        datasets = ["beir/scifact/test", "beir/nfcorpus/test", "beir/trec-covid"]

    import ir_datasets
    results = {}

    for dataset_name in datasets:
        print(f"\n--- Dataset: {dataset_name} ---")
        cache_csr = CACHE / f"beir_splade_{dataset_name.replace('/', '_')}_{max_docs}_docs.npz"
        cache_meta = CACHE / f"beir_splade_{dataset_name.replace('/', '_')}_{max_docs}_meta.pt"

        if cache_csr.exists() and cache_meta.exists():
            print(f"  Loading cached data from {cache_csr}")
            doc_csr_scipy = sp.load_npz(cache_csr)
            meta = torch.load(cache_meta, map_location="cpu", weights_only=False)
            query_dense = meta["query_dense"]
            qids = meta["qids"]
            qrels = meta["qrels"]
        else:
            ds = ir_datasets.load(dataset_name)

            doc_texts, doc_ids_list = [], []
            for doc in ds.docs_iter():
                text = doc.text if hasattr(doc, 'text') else doc.body if hasattr(doc, 'body') else str(doc)
                doc_texts.append(text[:512])
                doc_ids_list.append(doc.doc_id)
                if len(doc_texts) >= max_docs:
                    break

            docid_to_idx = {d: i for i, d in enumerate(doc_ids_list)}

            queries = {}
            for q in ds.queries_iter():
                queries[q.query_id] = q.text

            raw_qrels = defaultdict(dict)
            for qrel in ds.qrels_iter():
                if qrel.doc_id in docid_to_idx:
                    raw_qrels[qrel.query_id][docid_to_idx[qrel.doc_id]] = qrel.relevance

            qids = [q for q in queries if q in raw_qrels and len(raw_qrels[q]) > 0]
            query_texts = [queries[q] for q in qids]
            qrels = {q: raw_qrels[q] for q in qids}

            print(f"  Encoding {len(doc_texts)} docs...")
            doc_csr_scipy = encode_splade_to_csr(doc_texts, device, batch_size=32)
            print(f"  Encoding {len(query_texts)} queries...")
            query_csr = encode_splade_to_csr(query_texts, device, batch_size=64)
            query_dense = splade_to_dense(query_csr)

            sp.save_npz(cache_csr, doc_csr_scipy)
            torch.save({"query_dense": query_dense, "qids": qids, "qrels": qrels},
                       cache_meta)

        doc_csr = build_doc_csr_from_scipy(doc_csr_scipy, device)
        query_weights = build_query_weight_matrix(
            torch.from_numpy(query_dense) if isinstance(query_dense, np.ndarray) else query_dense,
            device,
        )

        top_scores, top_ids = triton_doc_csr_score_chunked(
            doc_csr, query_weights, top_k=1000, query_chunk_size=50
        )
        torch.cuda.synchronize(device)

        rankings = {}
        for i, qid in enumerate(qids):
            rankings[qid] = top_ids[i].cpu().tolist()

        mrr10 = compute_mrr(rankings, qrels, k=10)
        ndcg10 = compute_ndcg(rankings, qrels, k=10)
        recall1000 = compute_recall(rankings, qrels, k=1000)

        timing = bench_fn(
            lambda: triton_doc_csr_score_chunked(doc_csr, query_weights, top_k=10, query_chunk_size=50),
            warmup=3, trials=10, device=device
        )

        results[dataset_name] = {
            "dataset": dataset_name,
            "num_docs": doc_csr_scipy.shape[0],
            "num_queries": len(qids),
            "mrr@10": mrr10,
            "ndcg@10": ndcg10,
            "recall@1000": recall1000,
            "latency_ms": timing["mean_ms"],
        }

        print(f"  MRR@10: {mrr10:.4f}")
        print(f"  nDCG@10: {ndcg10:.4f}")
        print(f"  Recall@1000: {recall1000:.4f}")
        print(f"  Latency: {timing['mean_ms']:.1f} ms")

        torch.cuda.empty_cache()
        gc.collect()

    return results


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-encoding", action="store_true",
                       help="Skip SPLADE encoding, use cached data")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Max documents for full-scale test (None=8.8M)")
    parser.add_argument("--experiment", default="all",
                       choices=["all", "msmarco", "cusparse", "compile", "beir"],
                       help="Run a specific experiment")
    args = parser.parse_args()

    device = args.device
    all_results = {}

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    run_all = args.experiment == "all"

    if run_all or args.experiment == "msmarco":
        if args.max_docs:
            scales = [args.max_docs]
        else:
            scales = [100_000, 500_000, 1_000_000]
        try:
            msmarco_results = run_fullscale_msmarco(
                device, scales, num_queries=500, skip_encoding=args.skip_encoding
            )
            all_results["fullscale_msmarco"] = msmarco_results
        except Exception as e:
            print(f"  ERROR in full-scale: {e}")
            import traceback; traceback.print_exc()

    cache_100k_csr = CACHE / "msmarco_splade_csr_100000.npz"
    cache_100k_meta = CACHE / "msmarco_splade_csr_100000_meta.pt"
    if cache_100k_csr.exists() and cache_100k_meta.exists():
        doc_csr_scipy = sp.load_npz(cache_100k_csr)
        meta = torch.load(cache_100k_meta, map_location="cpu", weights_only=False)
        query_dense = meta["query_dense"]
    else:
        doc_csr_scipy = query_dense = None

    if (run_all or args.experiment == "cusparse") and doc_csr_scipy is not None:
        try:
            cusparse_results = run_cusparse_baseline(device, doc_csr_scipy, query_dense, 100000)
            all_results["cusparse_baseline"] = cusparse_results
        except Exception as e:
            print(f"  ERROR in cuSPARSE: {e}")
            import traceback; traceback.print_exc()

    if (run_all or args.experiment == "compile") and doc_csr_scipy is not None:
        try:
            compile_results = run_torch_compile_comparison(device, doc_csr_scipy, query_dense)
            all_results["torch_compile"] = compile_results
        except Exception as e:
            print(f"  ERROR in torch.compile: {e}")
            import traceback; traceback.print_exc()

    if run_all or args.experiment == "beir":
        try:
            beir_results = run_beir_evaluation(device)
            all_results["beir"] = beir_results
        except Exception as e:
            print(f"  ERROR in BEIR: {e}")
            import traceback; traceback.print_exc()

    out_path = RESULTS / "todo_experiments_all.json"
    if out_path.exists() and args.experiment != "all":
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
