"""
Seismic Multi-Threaded Benchmark at 8.8M MS MARCO.

Tests Seismic with 1, 4, 8, 16, 32 threads AND multiple query_cut values
to show the full quality-speed tradeoff surface.
"""

import os
import sys
import json
import time
import tarfile
import numpy as np

# Ensure a Seismic environment env is active.
# Paths are overridable via env vars; defaults are repo-relative.
# HEAP_FACTOR NOTE
# NOTE: heap_factor must lie in (0,1); Seismic's docs recommend [0.7, 1.0].
# Earlier runs used 10.0, which is outside the domain and effectively disables
# block skipping: it collapsed Recall@1000 from 0.953 to 0.738 and made query_cut
# inert. The library performs no range validation, so the value must be set with care.

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("SEISMIC_DATA", os.path.join(_REPO, "seismic_data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(_REPO, "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import seismic


def load_queries():
    """Load queries from tar.gz."""
    query_path = os.path.join(DATA_DIR, "queries.tar.gz")
    string_type = seismic.get_seismic_string()
    query_ids = []
    query_components = []
    query_values = []

    with tarfile.open(query_path, 'r:gz') as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None:
                continue
            for line in f:
                line = line.decode('utf-8') if isinstance(line, bytes) else line
                line = line.strip()
                if not line:
                    continue
                try:
                    q = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = str(q.get('id', q.get('_id', q.get('qid', len(query_ids)))))
                vector = q.get('vector', {})
                if isinstance(vector, dict) and len(vector) > 0:
                    tokens = list(vector.keys())
                    values = [float(v) for v in vector.values()]
                else:
                    continue
                query_ids.append(qid)
                query_components.append(np.array(tokens, dtype=string_type))
                query_values.append(np.array(values, dtype=np.float32))

    return query_ids, query_components, query_values


def load_qrels():
    """Load MS MARCO qrels."""
    qrels_path = os.path.join(DATA_DIR, "qrels.dev.small.tsv")
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
            elif len(parts) == 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
            else:
                continue
            if rel > 0:
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(did)
    return qrels


def evaluate_results(results_list, query_ids, qrels, k_mrr=10, k_recall=1000):
    """Compute MRR@10 and Recall@1000 from Seismic results.

    results_list is a list of lists of tuples (query_id, score, doc_id).
    Results may be reordered by Seismic, so we group by the query_id in each tuple.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for hits_for_query in results_list:
        if not hits_for_query:
            continue
        for hit in hits_for_query:
            if isinstance(hit, (list, tuple)) and len(hit) >= 3:
                qid_in_result, score, doc_id = str(hit[0]), hit[1], str(hit[2])
                grouped[qid_in_result].append((score, doc_id))

    for qid in grouped:
        grouped[qid].sort(key=lambda x: -x[0])

    mrr_scores = []
    recall_scores = []

    for qid in query_ids:
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        hits = grouped.get(qid, [])
        doc_ids = [did for _, did in hits]

        rr = 0.0
        for rank, did in enumerate(doc_ids[:k_mrr]):
            if did in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrr_scores.append(rr)

        retrieved = set(doc_ids[:k_recall])
        recall_scores.append(len(relevant & retrieved) / len(relevant) if relevant else 0.0)

    return {
        'mrr@10': float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        'recall@1000': float(np.mean(recall_scores)) if recall_scores else 0.0,
        'num_evaluated': len(mrr_scores),
    }


def main():
    # Load index
    index_path = os.path.join(DATA_DIR, "seismic_index.bin.index.seismic")
    print(f"Loading Seismic index from {index_path}...")
    index = seismic.SeismicIndex.load(index_path)
    n_docs = index.len
    print(f"  Index loaded: {n_docs:,} documents")

    # Load queries and qrels
    query_ids, query_components, query_values = load_queries()
    qrels = load_qrels()
    n_queries = len(query_ids)
    print(f"  {n_queries} queries loaded, {len(qrels)} with qrels")

    string_type = seismic.get_seismic_string()
    query_ids_arr = np.array(query_ids, dtype=string_type)

    # Warmup
    print("Warmup...")
    _ = index.batch_search(
        query_ids_arr[:100], query_components[:100], query_values[:100],
        k=10, query_cut=5, heap_factor=0.7, num_threads=4
    )

    all_results = {
        'system': 'Seismic',
        'n_docs': n_docs,
        'n_queries': n_queries,
    }

    # Test 1: Thread scaling at query_cut=5 (low quality, fast)
    print("\n" + "="*70)
    print("TEST 1: Thread scaling (query_cut=5, k=10)")
    print("="*70)

    thread_counts = [1, 2, 4, 8, 16, 32]
    thread_results = {}

    for n_threads in thread_counts:
        times = []
        for trial in range(3):
            t0 = time.perf_counter()
            results = index.batch_search(
                query_ids_arr, query_components, query_values,
                k=10, query_cut=5, heap_factor=0.7, num_threads=n_threads
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        mean_time = np.mean(times)
        per_query_us = mean_time / n_queries * 1e6
        qps = n_queries / mean_time

        thread_results[n_threads] = {
            'num_threads': n_threads,
            'total_time_s': float(mean_time),
            'per_query_us': float(per_query_us),
            'qps': float(qps),
            'times_s': [float(t) for t in times],
        }
        print(f"  {n_threads} threads: {per_query_us:.1f} μs/query, {qps:.0f} QPS "
              f"(total: {mean_time:.3f}s)")

    all_results['thread_scaling_qcut5'] = thread_results

    # Test 2: Quality-speed tradeoff (varying query_cut, 8 threads)
    print("\n" + "="*70)
    print("TEST 2: Quality-speed tradeoff (8 threads, k=1000)")
    print("="*70)

    query_cuts = [5, 10, 20, 30, 50]
    qcut_results = {}

    for qcut in query_cuts:
        # Measure latency
        t0 = time.perf_counter()
        results = index.batch_search(
            query_ids_arr, query_components, query_values,
            k=1000, query_cut=qcut, heap_factor=0.7, num_threads=8
        )
        elapsed = time.perf_counter() - t0
        per_query_us = elapsed / n_queries * 1e6

        # Measure quality
        metrics = evaluate_results(results, query_ids, qrels)

        qcut_results[qcut] = {
            'query_cut': qcut,
            'per_query_us': float(per_query_us),
            'total_time_s': float(elapsed),
            'qps': float(n_queries / elapsed),
            **metrics,
        }
        print(f"  query_cut={qcut}: {per_query_us:.1f} μs/query, "
              f"MRR@10={metrics['mrr@10']:.4f}, R@1000={metrics['recall@1000']:.4f}")

    all_results['quality_speed_tradeoff'] = qcut_results

    # Test 3: Best quality setting with thread scaling
    print("\n" + "="*70)
    print("TEST 3: High-quality (query_cut=50) thread scaling")
    print("="*70)

    hq_thread_results = {}
    for n_threads in [1, 4, 8, 16, 32]:
        t0 = time.perf_counter()
        results = index.batch_search(
            query_ids_arr, query_components, query_values,
            k=1000, query_cut=50, heap_factor=0.7, num_threads=n_threads
        )
        elapsed = time.perf_counter() - t0
        per_query_us = elapsed / n_queries * 1e6
        metrics = evaluate_results(results, query_ids, qrels)

        hq_thread_results[n_threads] = {
            'num_threads': n_threads,
            'per_query_us': float(per_query_us),
            'qps': float(n_queries / elapsed),
            **metrics,
        }
        print(f"  {n_threads} threads: {per_query_us:.1f} μs/query, "
              f"MRR@10={metrics['mrr@10']:.4f}, R@1000={metrics['recall@1000']:.4f}")

    all_results['high_quality_thread_scaling'] = hq_thread_results

    # Save
    output_path = os.path.join(RESULTS_DIR, "seismic_multithread_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
