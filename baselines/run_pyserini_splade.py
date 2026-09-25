"""
Pyserini SPLADE CPU Baseline at 8.8M MS MARCO.

Uses pre-built Lucene impact index for SPLADE-pp-ed (exact scoring on CPU).
This gives us the ground truth MRR@10. Our GPU system must match it exactly.
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

# Output dir. Override with RESULTS_DIR env var; defaults to ../results relative to this file.
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_msmarco_queries():
    """Load MS MARCO dev-small queries via ir_datasets."""
    import ir_datasets
    ds = ir_datasets.load("msmarco-passage/dev/small")

    queries = {}
    for q in ds.queries_iter():
        queries[q.query_id] = q.text

    qrels = defaultdict(dict)
    for qrel in ds.qrels_iter():
        if qrel.relevance > 0:
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    valid_qids = [q for q in queries if q in qrels]
    print(f"Loaded {len(queries)} queries, {len(valid_qids)} with qrels")
    return queries, qrels, valid_qids


def main():
    from pyserini.search.lucene import LuceneImpactSearcher

    print("="*70)
    print("Pyserini SPLADE CPU Baseline (8.8M MS MARCO)")
    print("="*70)

    # Load queries
    queries, qrels, valid_qids = load_msmarco_queries()

    # Load pre-built SPLADE index
    print("\nLoading Pyserini SPLADE index (will download if needed)...")
    t0 = time.time()
    searcher = LuceneImpactSearcher.from_prebuilt_index(
        'msmarco-v1-passage.splade-pp-ed',
        query_encoder='naver/splade-cocondenser-ensembledistil'
    )
    load_time = time.time() - t0
    print(f"  Index loaded in {load_time:.1f}s")
    print(f"  Total docs: {searcher.num_docs}")

    all_results = {
        'system': 'Pyserini_SPLADE',
        'index': 'msmarco-v1-passage.splade-pp-ed',
        'encoder': 'SpladePlusPlusEnsembleDistil',
        'num_docs': searcher.num_docs,
        'num_queries': len(valid_qids),
    }

    # Single-threaded retrieval with timing
    print(f"\nRunning retrieval ({len(valid_qids)} queries, single-threaded)...")
    rankings = {}
    latencies = []

    for i, qid in enumerate(valid_qids):
        query_text = queries[qid]
        t0 = time.perf_counter()
        hits = searcher.search(query_text, k=1000)
        elapsed = (time.perf_counter() - t0) * 1000  # ms

        latencies.append(elapsed)
        rankings[qid] = [hit.docid for hit in hits]

        if (i + 1) % 1000 == 0:
            avg_lat = np.mean(latencies[-1000:])
            print(f"  {i+1}/{len(valid_qids)} queries, avg latency: {avg_lat:.1f} ms")

    # Compute metrics
    mrr_scores = []
    ndcg_scores = []
    recall_scores = []

    from math import log2
    for qid in valid_qids:
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        ranked = rankings.get(qid, [])

        # MRR@10
        rr = 0.0
        for rank, did in enumerate(ranked[:10]):
            if did in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrr_scores.append(rr)

        # nDCG@10
        dcg = 0.0
        for rank, did in enumerate(ranked[:10]):
            if did in relevant:
                dcg += relevant[did] / log2(rank + 2)
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum(r / log2(i + 2) for i, r in enumerate(ideal))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # Recall@1000
        rel_set = set(relevant.keys())
        retrieved = set(ranked[:1000])
        recall_scores.append(len(rel_set & retrieved) / len(rel_set) if rel_set else 0.0)

    metrics = {
        'mrr@10': float(np.mean(mrr_scores)),
        'ndcg@10': float(np.mean(ndcg_scores)),
        'recall@1000': float(np.mean(recall_scores)),
        'num_evaluated': len(mrr_scores),
    }

    latency_stats = {
        'mean_ms': float(np.mean(latencies)),
        'median_ms': float(np.median(latencies)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'total_s': float(sum(latencies) / 1000),
        'qps': float(len(latencies) / (sum(latencies) / 1000)),
    }

    all_results['metrics'] = metrics
    all_results['latency'] = latency_stats

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  MRR@10: {metrics['mrr@10']:.4f}")
    print(f"  nDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"  Recall@1000: {metrics['recall@1000']:.4f}")
    print(f"  Mean latency: {latency_stats['mean_ms']:.1f} ms/query")
    print(f"  Median latency: {latency_stats['median_ms']:.1f} ms/query")
    print(f"  Throughput: {latency_stats['qps']:.1f} QPS")

    # Save
    output_path = os.path.join(RESULTS_DIR, "pyserini_splade_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save rankings for comparison with GPU system
    rankings_path = os.path.join(RESULTS_DIR, "pyserini_splade_rankings.json")
    with open(rankings_path, 'w') as f:
        json.dump(rankings, f)
    print(f"Rankings saved to {rankings_path}")


if __name__ == "__main__":
    main()
