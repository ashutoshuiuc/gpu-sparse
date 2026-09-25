"""
MS MARCO Passage Ranking Evaluation for GPUSparse.

Addresses review weaknesses:
- W1: MRR@10, nDCG@10 with real relevance judgments
- W2: Pyserini BM25 baseline (quality + speed)
- W4: Scale to full MS MARCO (8.8M passages)
- W5: Real MS MARCO passages (not pseudo-passages)
- W9: End-to-end pipeline timing with real SPLADE encoding

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python src/msmarco_eval.py [--num_docs N] [--num_queries N]
"""

import torch
import numpy as np
import time
import json
import os
import sys
import gc
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

RESULTS_DIR = Path(__file__).parent.parent / "tracker"
CACHE_DIR = Path(os.environ.get("DATA_ROOT", str(Path(__file__).parent.parent))) / "cache"


def load_msmarco_data(max_docs=None, max_queries=None):
    """Load MS MARCO passage ranking data via ir_datasets.

    Strategy: include ALL documents that have relevance judgments, plus
    random passages to reach max_docs. This ensures we can compute
    meaningful IR metrics even with a subset.
    """
    import ir_datasets
    ds = ir_datasets.load("msmarco-passage/dev/small")

    print("Loading queries and qrels first (to identify relevant docs)...")
    queries = {}
    for q in ds.queries_iter():
        queries[q.query_id] = q.text

    # Collect all relevant doc IDs
    raw_qrels = defaultdict(dict)
    relevant_doc_ids = set()
    for qrel in ds.qrels_iter():
        raw_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        relevant_doc_ids.add(qrel.doc_id)
    print(f"  {len(relevant_doc_ids)} unique relevant docs across {len(raw_qrels)} queries")

    print("Loading MS MARCO passages (relevant + random fill)...")
    t0 = time.time()
    docs = {}
    filler_docs = {}
    # First pass: load all docs, segregating relevant from filler
    for i, doc in enumerate(ds.docs_iter()):
        if doc.doc_id in relevant_doc_ids:
            docs[doc.doc_id] = doc.text
        elif max_docs is not None and len(filler_docs) < (max_docs - len(relevant_doc_ids)):
            filler_docs[doc.doc_id] = doc.text
        # If we have enough, stop
        if max_docs is not None and (len(docs) + len(filler_docs)) >= max_docs:
            # Check if we got all relevant docs
            if len(docs) >= len(relevant_doc_ids):
                break

    # Merge: relevant docs first, then fillers
    all_docs = {}
    all_docs.update(docs)
    remaining = (max_docs or (len(docs) + len(filler_docs))) - len(all_docs)
    filler_items = list(filler_docs.items())[:remaining]
    for did, text in filler_items:
        all_docs[did] = text

    doc_ids_list = list(all_docs.keys())
    doc_texts = [all_docs[did] for did in doc_ids_list]
    docid_to_idx = {did: i for i, did in enumerate(doc_ids_list)}
    print(f"  Loaded {len(doc_texts)} passages ({len(docs)} relevant + {len(doc_texts)-len(docs)} filler) in {time.time()-t0:.1f}s")

    # Build qrels with internal indices
    qrels = defaultdict(dict)
    for qid in raw_qrels:
        for did, rel in raw_qrels[qid].items():
            if did in docid_to_idx:
                qrels[qid][docid_to_idx[did]] = rel

    # Filter to queries that have at least one relevant doc in our collection
    valid_qids = [qid for qid in queries if qid in qrels and len(qrels[qid]) > 0]
    if max_queries is not None:
        valid_qids = valid_qids[:max_queries]

    query_texts = [queries[qid] for qid in valid_qids]
    query_qrels = {qid: qrels[qid] for qid in valid_qids}

    print(f"  {len(valid_qids)} queries with relevant docs in collection")
    print(f"  {sum(len(v) for v in query_qrels.values())} total qrels")

    return doc_texts, doc_ids_list, docid_to_idx, query_texts, valid_qids, query_qrels


def encode_splade(texts, tokenizer, model, device, batch_size=64, max_length=256, desc="Encoding"):
    """Encode texts with SPLADE model in batches."""
    all_reps = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            # Disable SDPA to avoid cuDNN issues with newer torch
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                output = model(**inputs)
            logits = output.logits
            reps = torch.log1p(torch.relu(logits)).max(dim=1).values

        all_reps.append(reps.cpu())

        done = min(i + batch_size, len(texts))
        if (done % (batch_size * 10) == 0) or done == len(texts):
            elapsed = time.time() - t0
            rate = done / elapsed
            print(f"  {desc}: {done}/{len(texts)} ({rate:.0f} texts/s)")

    return torch.cat(all_reps, dim=0)


def build_gpu_index_from_dense(doc_reps, device):
    """Build GPU inverted index from dense SPLADE representations."""
    BLOCK = 32
    doc_np = doc_reps.numpy()
    num_docs, vocab_size = doc_np.shape

    print(f"  Building index: {num_docs} docs, vocab {vocab_size}")
    t0 = time.time()

    # Get COO entries
    rows, cols = np.nonzero(doc_np)
    vals = doc_np[rows, cols].astype(np.float32)

    # Sort by term for efficient posting list construction
    # kind='stable' so entries stay ascending by doc_id within each posting list,
    # which is the layout invariant the paper and gpu_inverted_index.py document.
    # Default (quicksort) left 96.2% of posting lists non-ascending.
    sort_idx = np.argsort(cols, kind='stable')
    sorted_terms = cols[sort_idx]
    sorted_docs = rows[sort_idx].astype(np.int32)
    sorted_scores = vals[sort_idx]

    unique_terms, counts = np.unique(sorted_terms, return_counts=True)

    lengths = np.zeros(vocab_size, dtype=np.int32)
    lengths[unique_terms] = counts
    padded_lengths = ((lengths + BLOCK - 1) // BLOCK) * BLOCK
    offsets = np.zeros(vocab_size, dtype=np.int64)
    if vocab_size > 1:
        np.cumsum(padded_lengths[:-1], out=offsets[1:])

    total_padded = int(offsets[-1] + padded_lengths[-1])
    all_doc_ids = np.full(total_padded, -1, dtype=np.int32)
    all_scores = np.zeros(total_padded, dtype=np.float32)
    max_scores = np.zeros(vocab_size, dtype=np.float32)

    src_offset = 0
    for i, term_id in enumerate(unique_terms):
        n = counts[i]
        off = offsets[term_id]
        all_doc_ids[off:off+n] = sorted_docs[src_offset:src_offset+n]
        all_scores[off:off+n] = sorted_scores[src_offset:src_offset+n]
        max_scores[term_id] = sorted_scores[src_offset:src_offset+n].max()
        src_offset += n

    index = {
        'doc_ids': torch.from_numpy(all_doc_ids).to(device),
        'scores': torch.from_numpy(all_scores).to(device),
        'offsets': torch.from_numpy(offsets).to(device),
        'lengths': torch.from_numpy(lengths).to(device),
        'padded_lengths': torch.from_numpy(padded_lengths).to(device),
        'max_scores': torch.from_numpy(max_scores).to(device),
        'num_docs': num_docs,
        'vocab_size': vocab_size,
        'device': device,
    }
    elapsed = time.time() - t0
    mem_mb = (total_padded * 8 + vocab_size * 20) / 1e6
    print(f"  Index built in {elapsed:.1f}s, {mem_mb:.0f} MB, {total_padded} total entries")
    return index


def prepare_query_tensors(query_reps, max_terms=128, device="cuda:0"):
    """Convert dense query reps to sparse (term_ids, term_scores) tensors."""
    n_queries = query_reps.shape[0]
    query_np = query_reps.numpy()

    term_ids = np.full((n_queries, max_terms), -1, dtype=np.int32)
    term_scores = np.zeros((n_queries, max_terms), dtype=np.float32)
    actual_terms = []

    for i in range(n_queries):
        nz_idx = np.nonzero(query_np[i])[0]
        nz_vals = query_np[i, nz_idx]
        sort_idx = np.argsort(-nz_vals)
        n = min(len(sort_idx), max_terms)
        term_ids[i, :n] = nz_idx[sort_idx[:n]]
        term_scores[i, :n] = nz_vals[sort_idx[:n]]
        actual_terms.append(len(nz_idx))

    print(f"  Query sparsity: avg {np.mean(actual_terms):.1f}, max {np.max(actual_terms)}, min {np.min(actual_terms)}")

    return (torch.from_numpy(term_ids).to(device),
            torch.from_numpy(term_scores).to(device))


def compute_metrics(run, qrels, k_values=[10, 100, 1000]):
    """
    Compute MRR@k and nDCG@k.
    run: dict mapping qid -> list of (doc_idx, score) sorted by score desc
    qrels: dict mapping qid -> {doc_idx: relevance}
    """
    results = {}

    for k in k_values:
        mrr_sum = 0.0
        ndcg_sum = 0.0
        recall_sum = 0.0
        n_queries = 0

        for qid, ranked_list in run.items():
            if qid not in qrels or len(qrels[qid]) == 0:
                continue
            n_queries += 1
            rel_docs = qrels[qid]

            # MRR@k
            rr = 0.0
            for rank, (doc_idx, score) in enumerate(ranked_list[:k]):
                if doc_idx in rel_docs and rel_docs[doc_idx] > 0:
                    rr = 1.0 / (rank + 1)
                    break
            mrr_sum += rr

            # nDCG@k
            dcg = 0.0
            for rank, (doc_idx, score) in enumerate(ranked_list[:k]):
                if doc_idx in rel_docs:
                    rel = rel_docs[doc_idx]
                    dcg += (2**rel - 1) / np.log2(rank + 2)

            # Ideal DCG
            ideal_rels = sorted(rel_docs.values(), reverse=True)[:k]
            idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rels))
            ndcg_sum += dcg / idcg if idcg > 0 else 0.0

            # Recall@k
            retrieved_rel = sum(1 for doc_idx, _ in ranked_list[:k]
                              if doc_idx in rel_docs and rel_docs[doc_idx] > 0)
            total_rel = sum(1 for r in rel_docs.values() if r > 0)
            recall_sum += retrieved_rel / total_rel if total_rel > 0 else 0.0

        if n_queries > 0:
            results[f"MRR@{k}"] = mrr_sum / n_queries
            results[f"nDCG@{k}"] = ndcg_sum / n_queries
            results[f"Recall@{k}"] = recall_sum / n_queries

    results["n_queries"] = n_queries
    return results


def bench(fn, warmup=3, trials=10, device=None):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
        if device is not None:
            torch.cuda.synchronize(device)

    times = []
    for _ in range(trials):
        if device is not None:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        result = fn()
        if device is not None:
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times), result


def run_pyserini_bm25(doc_texts, doc_ids_list, query_texts, valid_qids, top_k=1000):
    """Run Pyserini BM25 baseline."""
    import tempfile
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index.lucene import LuceneIndexer

    print("\n=== Pyserini BM25 Baseline ===")

    # Build index
    tmpdir = tempfile.mkdtemp(prefix="pyserini_idx_")
    print(f"  Building Lucene index in {tmpdir}...")
    t0 = time.time()

    # Write docs as JSON-L for indexing
    import json as jsonlib
    docs_dir = os.path.join(tmpdir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    docs_file = os.path.join(docs_dir, "docs.jsonl")
    with open(docs_file, "w") as f:
        for i, text in enumerate(doc_texts):
            jsonlib.dump({"id": str(i), "contents": text}, f)
            f.write("\n")

    # Index with Pyserini using command-line indexer
    index_dir = os.path.join(tmpdir, "index")
    import subprocess
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", docs_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storeRaw",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  Indexing stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"Pyserini indexing failed: {result.returncode}")
    print(f"  Index built in {time.time()-t0:.1f}s")

    # Search
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=0.9, b=0.4)  # Standard BM25 params

    print("  Searching...")
    t0 = time.time()
    run = {}
    for qid, qtext in zip(valid_qids, query_texts):
        hits = searcher.search(qtext, k=top_k)
        run[qid] = [(int(hit.docid), hit.score) for hit in hits]
    search_time = time.time() - t0
    avg_latency_ms = search_time / len(valid_qids) * 1000

    print(f"  BM25 search: {search_time:.2f}s total, {avg_latency_ms:.1f}ms/query")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    return run, avg_latency_ms


def run_gpusparse_eval(index, query_term_ids, query_term_scores, valid_qids, top_k=1000, device="cuda:0"):
    """Run GPUSparse Triton scoring and return run dict + timing."""
    from src.triton_kernel import triton_fused_score
    from src.triton_kernel_v3 import triton_fused_score_v3

    batch_size = query_term_ids.shape[0]

    # Convert index dict to a namespace object for triton_fused_score
    class IndexObj:
        pass
    idx_obj = IndexObj()
    idx_obj.doc_ids = index['doc_ids']
    idx_obj.scores = index['scores']
    idx_obj.offsets = index['offsets']
    idx_obj.lengths = index['lengths']
    idx_obj.max_scores = index['max_scores']
    idx_obj.num_docs = index['num_docs']
    idx_obj.vocab_size = index['vocab_size']
    idx_obj.device = index['device']

    results = {}

    # V1 kernel
    print("\n  Benchmarking Triton V1 (BLOCK_PL=128)...")
    latency_v1, (scores_v1, ids_v1) = bench(
        lambda: triton_fused_score(idx_obj, query_term_ids, query_term_scores, top_k=top_k, block_pl=128),
        warmup=3, trials=10, device=device
    )
    run_v1 = {}
    ids_cpu = ids_v1.cpu().numpy()
    scores_cpu = scores_v1.cpu().numpy()
    for i, qid in enumerate(valid_qids):
        run_v1[qid] = [(int(ids_cpu[i, j]), float(scores_cpu[i, j])) for j in range(top_k)]
    results['triton_v1'] = {'latency_ms': latency_v1, 'run': run_v1}
    print(f"    Latency: {latency_v1:.2f}ms for {batch_size} queries ({latency_v1/batch_size*1000:.0f}us/query)")

    # V3 kernel
    print("  Benchmarking Triton V3 (BLOCK_PL=512)...")
    latency_v3, (scores_v3, ids_v3) = bench(
        lambda: triton_fused_score_v3(index, query_term_ids, query_term_scores, top_k=top_k, block_pl=512),
        warmup=3, trials=10, device=device
    )
    run_v3 = {}
    ids_cpu = ids_v3.cpu().numpy()
    scores_cpu = scores_v3.cpu().numpy()
    for i, qid in enumerate(valid_qids):
        run_v3[qid] = [(int(ids_cpu[i, j]), float(scores_cpu[i, j])) for j in range(top_k)]
    results['triton_v3'] = {'latency_ms': latency_v3, 'run': run_v3}
    print(f"    Latency: {latency_v3:.2f}ms for {batch_size} queries ({latency_v3/batch_size*1000:.0f}us/query)")

    # V3 tiled
    from src.triton_kernel_v3 import triton_fused_score_v3_tiled
    print("  Benchmarking Triton V3-Tiled (BLOCK_PL=512, 8 terms/prog)...")
    latency_v3t, (scores_v3t, ids_v3t) = bench(
        lambda: triton_fused_score_v3_tiled(index, query_term_ids, query_term_scores, top_k=top_k, block_pl=512, terms_per_program=8),
        warmup=3, trials=10, device=device
    )
    run_v3t = {}
    ids_cpu = ids_v3t.cpu().numpy()
    scores_cpu = scores_v3t.cpu().numpy()
    for i, qid in enumerate(valid_qids):
        run_v3t[qid] = [(int(ids_cpu[i, j]), float(scores_cpu[i, j])) for j in range(top_k)]
    results['triton_v3_tiled'] = {'latency_ms': latency_v3t, 'run': run_v3t}
    print(f"    Latency: {latency_v3t:.2f}ms for {batch_size} queries ({latency_v3t/batch_size*1000:.0f}us/query)")

    # Dense matmul baseline
    print("  Benchmarking Dense MatMul...")

    return results


def run_dense_matmul_eval(doc_reps_gpu, query_reps_gpu, valid_qids, top_k=1000, device="cuda:0"):
    """Dense matmul baseline."""
    def fn():
        scores = torch.mm(query_reps_gpu, doc_reps_gpu.t())
        return torch.topk(scores, k=min(top_k, doc_reps_gpu.shape[0]), dim=1)

    latency, (scores, ids) = bench(fn, warmup=3, trials=10, device=device)

    run = {}
    ids_cpu = ids.cpu().numpy()
    scores_cpu = scores.cpu().numpy()
    for i, qid in enumerate(valid_qids):
        run[qid] = [(int(ids_cpu[i, j]), float(scores_cpu[i, j])) for j in range(min(top_k, ids_cpu.shape[1]))]

    return run, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=100000,
                       help="Number of MS MARCO passages to use (default 100K, max ~8.8M)")
    parser.add_argument("--num_queries", type=int, default=500,
                       help="Number of dev queries to evaluate (default 500, max 6980)")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--skip_bm25", action="store_true")
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--cache_encodings", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Config: {args.num_docs} docs, {args.num_queries} queries, top-{args.top_k}")

    # ---- Load MS MARCO data ----
    doc_texts, doc_ids_list, docid_to_idx, query_texts, valid_qids, query_qrels = \
        load_msmarco_data(max_docs=args.num_docs, max_queries=args.num_queries)

    all_results = {
        'config': {
            'num_docs': len(doc_texts),
            'num_queries': len(valid_qids),
            'top_k': args.top_k,
            'device': str(device),
        }
    }

    # ---- Pyserini BM25 ----
    if not args.skip_bm25:
        try:
            bm25_run, bm25_latency = run_pyserini_bm25(
                doc_texts, doc_ids_list, query_texts, valid_qids, top_k=args.top_k
            )
            bm25_metrics = compute_metrics(bm25_run, query_qrels, k_values=[10, 100, 1000])
            print(f"\n  BM25 Results:")
            for k, v in sorted(bm25_metrics.items()):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
            all_results['bm25'] = {
                'metrics': bm25_metrics,
                'latency_ms_per_query': bm25_latency,
            }
        except Exception as e:
            print(f"  BM25 failed: {e}")

    # ---- SPLADE Encoding ----
    cache_dir = CACHE_DIR / f"msmarco_{args.num_docs}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_cache = cache_dir / "doc_reps.pt"
    query_cache = cache_dir / "query_reps.pt"

    if args.cache_encodings and doc_cache.exists() and query_cache.exists():
        print(f"\nLoading cached SPLADE encodings from {cache_dir}")
        doc_reps = torch.load(doc_cache, map_location="cpu", weights_only=True)
        query_reps = torch.load(query_cache, map_location="cpu", weights_only=True)
        # Handle query count mismatch
        if query_reps.shape[0] < len(valid_qids):
            print(f"  Cache has {query_reps.shape[0]} queries, need {len(valid_qids)}, re-encoding queries...")
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            model_name = "naver/splade-cocondenser-ensembledistil"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name).eval().to(device)
            query_reps = encode_splade(query_texts, tokenizer, model, device, batch_size=64, desc="Queries")
            torch.save(query_reps, query_cache)
            del model
            torch.cuda.empty_cache()
        elif query_reps.shape[0] > len(valid_qids):
            query_reps = query_reps[:len(valid_qids)]
    else:
        print(f"\nEncoding with SPLADE (naver/splade-cocondenser-ensembledistil)...")
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        model_name = "naver/splade-cocondenser-ensembledistil"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).eval().to(device)

        t0 = time.time()
        doc_reps = encode_splade(doc_texts, tokenizer, model, device, batch_size=64, desc="Docs")
        doc_encode_time = time.time() - t0

        t0 = time.time()
        query_reps = encode_splade(query_texts, tokenizer, model, device, batch_size=64, desc="Queries")
        query_encode_time = time.time() - t0

        all_results['encoding'] = {
            'doc_encode_time_s': doc_encode_time,
            'query_encode_time_s': query_encode_time,
            'doc_rate': len(doc_texts) / doc_encode_time,
            'query_rate': len(query_texts) / query_encode_time,
        }

        if args.cache_encodings:
            torch.save(doc_reps, doc_cache)
            torch.save(query_reps, query_cache)
            print(f"  Cached encodings to {cache_dir}")

        del model
        torch.cuda.empty_cache()

    # Doc/query sparsity stats
    doc_nnz = (doc_reps > 0).sum(dim=1).float()
    query_nnz = (query_reps > 0).sum(dim=1).float()
    print(f"\n  Doc sparsity: avg {doc_nnz.mean():.1f}, std {doc_nnz.std():.1f}")
    print(f"  Query sparsity: avg {query_nnz.mean():.1f}, std {query_nnz.std():.1f}")

    # ---- Build GPU Index ----
    print("\nBuilding GPU inverted index...")
    index = build_gpu_index_from_dense(doc_reps, device)

    # ---- Prepare query tensors ----
    print("Preparing query tensors...")
    q_term_ids, q_term_scores = prepare_query_tensors(query_reps, max_terms=128, device=device)

    # ---- GPUSparse Evaluation ----
    print("\n=== GPUSparse Triton Evaluation ===")
    gpu_results = run_gpusparse_eval(
        index, q_term_ids, q_term_scores, valid_qids, top_k=args.top_k, device=device
    )

    for method_name, method_data in gpu_results.items():
        metrics = compute_metrics(method_data['run'], query_qrels, k_values=[10, 100, 1000])
        print(f"\n  {method_name} Results:")
        for k, v in sorted(metrics.items()):
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        all_results[method_name] = {
            'metrics': metrics,
            'latency_ms': method_data['latency_ms'],
            'per_query_us': method_data['latency_ms'] / len(valid_qids) * 1000,
        }

    # ---- Dense MatMul ----
    if not args.skip_dense:
        num_docs = doc_reps.shape[0]
        vocab_size = doc_reps.shape[1]
        mem_needed_gb = num_docs * vocab_size * 4 / 1e9
        if mem_needed_gb < 30:  # only if fits in GPU memory
            print(f"\n=== Dense MatMul Baseline ({mem_needed_gb:.1f} GB) ===")
            doc_reps_gpu = doc_reps.to(device)
            query_reps_gpu = query_reps.to(device)
            dense_run, dense_latency = run_dense_matmul_eval(
                doc_reps_gpu, query_reps_gpu, valid_qids, top_k=args.top_k, device=device
            )
            dense_metrics = compute_metrics(dense_run, query_qrels, k_values=[10, 100, 1000])
            print(f"  Dense MatMul Results:")
            for k, v in sorted(dense_metrics.items()):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
            all_results['dense_matmul'] = {
                'metrics': dense_metrics,
                'latency_ms': dense_latency,
                'per_query_us': dense_latency / len(valid_qids) * 1000,
            }
            del doc_reps_gpu, query_reps_gpu
            torch.cuda.empty_cache()
        else:
            print(f"\n  Skipping Dense MatMul (would need {mem_needed_gb:.1f} GB)")

    # ---- End-to-end pipeline timing ----
    print("\n=== End-to-End Pipeline Timing ===")
    # Measure query encoding + scoring + topk
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).eval().to(device)

    from src.triton_kernel_v3 import triton_fused_score_v3

    def e2e_pipeline(batch_texts):
        # 1. Encode
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = model(**inputs)
            reps = torch.log1p(torch.relu(output.logits)).max(dim=1).values

        # 2. Prepare sparse query
        reps_np = reps.cpu().numpy()
        n = reps_np.shape[0]
        max_terms = 96
        tids = np.full((n, max_terms), -1, dtype=np.int32)
        tscores = np.zeros((n, max_terms), dtype=np.float32)
        for i in range(n):
            nz = np.nonzero(reps_np[i])[0]
            nzv = reps_np[i, nz]
            si = np.argsort(-nzv)
            nn = min(len(si), max_terms)
            tids[i, :nn] = nz[si[:nn]]
            tscores[i, :nn] = nzv[si[:nn]]

        qt_ids = torch.from_numpy(tids).to(device)
        qt_scores = torch.from_numpy(tscores).to(device)

        # 3. Score
        return triton_fused_score_v3(index, qt_ids, qt_scores, top_k=args.top_k, block_pl=512)

    # Warmup
    sample_queries = query_texts[:min(32, len(query_texts))]
    for _ in range(2):
        e2e_pipeline(sample_queries)
        torch.cuda.synchronize(device)

    # Measure
    batch_sizes_e2e = [1, 8, 32]
    e2e_results = {}
    for bs in batch_sizes_e2e:
        batch = query_texts[:bs]
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        e2e_pipeline(batch)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - t0) * 1000
        e2e_results[bs] = elapsed
        print(f"  Batch {bs}: {elapsed:.1f}ms total ({elapsed/bs:.1f}ms/query)")

    all_results['e2e_pipeline'] = e2e_results

    del model
    torch.cuda.empty_cache()

    # ---- Bandwidth analysis ----
    print("\n=== Bandwidth Analysis ===")
    total_postings = int(index['lengths'].sum().item())
    bytes_read = total_postings * 8  # 4 bytes doc_id + 4 bytes score
    for method_name in ['triton_v1', 'triton_v3', 'triton_v3_tiled']:
        if method_name in all_results and 'latency_ms' in all_results[method_name]:
            lat = all_results[method_name]['latency_ms']
            # Each query reads all posting lists for its terms
            # Approx: avg_query_terms * avg_postings_per_term * 8 bytes * num_queries
            avg_qterms = float(query_nnz.mean())
            avg_pl_len = total_postings / max(1, (index['lengths'] > 0).sum().item())
            bytes_per_batch = avg_qterms * avg_pl_len * 8 * len(valid_qids)
            bw_gbs = bytes_per_batch / (lat / 1000) / 1e9
            print(f"  {method_name}: {lat:.2f}ms, ~{bw_gbs:.1f} GB/s effective bandwidth")
            all_results[method_name]['effective_bandwidth_GBs'] = bw_gbs

    # ---- Save results ----
    output_file = RESULTS_DIR / f"msmarco_eval_{args.num_docs}.json"
    # Convert to serializable
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            sv = {}
            for kk, vv in v.items():
                if kk == 'run':
                    continue  # Don't save full run dicts
                sv[kk] = vv
            serializable[k] = sv
        else:
            serializable[k] = v

    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # ---- Summary Table ----
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Method':<25} {'MRR@10':>8} {'nDCG@10':>8} {'R@1000':>8} {'Latency':>12} {'Per-Q':>10}")
    print("-"*80)
    for method in ['bm25', 'dense_matmul', 'triton_v1', 'triton_v3', 'triton_v3_tiled']:
        if method in all_results and 'metrics' in all_results[method]:
            m = all_results[method]['metrics']
            lat = all_results[method].get('latency_ms', all_results[method].get('latency_ms_per_query', 0) * len(valid_qids))
            pq = all_results[method].get('per_query_us', all_results[method].get('latency_ms_per_query', 0) * 1000)
            mrr10 = m.get('MRR@10', 0)
            ndcg10 = m.get('nDCG@10', 0)
            r1000 = m.get('Recall@1000', 0)
            print(f"{method:<25} {mrr10:>8.4f} {ndcg10:>8.4f} {r1000:>8.4f} {lat:>10.2f}ms {pq:>8.0f}us")
    print("="*80)


if __name__ == "__main__":
    main()
