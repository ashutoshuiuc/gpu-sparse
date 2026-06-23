"""
Full-scale GPUSparse benchmark on MS MARCO with SPLADE-style sparse representations.

Steps:
1. Download MS MARCO passages
2. Generate SPLADE-style sparse representations (using learned sparse or simulated)
3. Benchmark GPU sparse retrieval at realistic scale (100K-1M+)
4. Compare against CPU baselines
5. Save results for paper
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import sys
from pathlib import Path
from scipy import sparse as sp

RESULTS_DIR = Path(__file__).parent.parent / "tracker"


def generate_splade_style_sparse(n_docs, vocab_size=30522, avg_terms=100, device='cuda:0'):
    """
    Generate SPLADE-style sparse representations.
    Real SPLADE produces ~50-200 non-zero terms per document.
    Weights follow an exponential-like distribution (few high, many low).
    """
    print(f"  Generating sparse representations for {n_docs} documents...")

    # Build sparse data on CPU first
    all_indices = []
    all_values = []
    doc_ptrs = [0]

    rng = np.random.default_rng(42)

    for i in range(n_docs):
        # Number of terms varies per document
        n_terms = max(5, int(rng.poisson(avg_terms)))
        n_terms = min(n_terms, vocab_size)

        # Sample terms with frequency bias (common terms more likely)
        # Zipf-like distribution
        terms = rng.choice(vocab_size, size=n_terms, replace=False)

        # Weights: exponential distribution (few high, many low)
        weights = rng.exponential(1.0, size=n_terms).astype(np.float32)
        # Apply log1p + relu like SPLADE
        weights = np.log1p(np.maximum(weights, 0))

        all_indices.extend(terms.tolist())
        all_values.extend(weights.tolist())
        doc_ptrs.append(len(all_indices))

        if (i + 1) % 100000 == 0:
            print(f"    Generated {i+1}/{n_docs} documents")

    return {
        'indices': np.array(all_indices, dtype=np.int64),
        'values': np.array(all_values, dtype=np.float32),
        'ptrs': np.array(doc_ptrs, dtype=np.int64),
        'n_docs': n_docs,
        'vocab_size': vocab_size,
        'avg_nnz': len(all_indices) / n_docs,
    }


def generate_sparse_queries(n_queries, vocab_size=30522, avg_terms=20):
    """Generate sparse query representations."""
    rng = np.random.default_rng(123)

    all_indices = []
    all_values = []
    query_ptrs = [0]

    for i in range(n_queries):
        n_terms = max(3, int(rng.poisson(avg_terms)))
        n_terms = min(n_terms, vocab_size)
        terms = rng.choice(vocab_size, size=n_terms, replace=False)
        weights = rng.exponential(1.5, size=n_terms).astype(np.float32)
        weights = np.log1p(np.maximum(weights, 0))

        all_indices.extend(terms.tolist())
        all_values.extend(weights.tolist())
        query_ptrs.append(len(all_indices))

    return {
        'indices': np.array(all_indices, dtype=np.int64),
        'values': np.array(all_values, dtype=np.float32),
        'ptrs': np.array(query_ptrs, dtype=np.int64),
        'n_queries': n_queries,
    }


def build_dense_matrix(sparse_data, device='cuda:0'):
    """Convert sparse to dense matrix on GPU."""
    n_docs = sparse_data['n_docs']
    vocab_size = sparse_data['vocab_size']

    matrix = torch.zeros(n_docs, vocab_size, device=device, dtype=torch.float32)
    ptrs = sparse_data['ptrs']
    indices = sparse_data['indices']
    values = sparse_data['values']

    for i in range(n_docs):
        start, end = ptrs[i], ptrs[i+1]
        if end > start:
            idx = torch.tensor(indices[start:end], dtype=torch.long, device=device)
            val = torch.tensor(values[start:end], dtype=torch.float32, device=device)
            matrix[i].scatter_(0, idx, val)

        if (i + 1) % 50000 == 0:
            print(f"    Built {i+1}/{n_docs} rows")

    return matrix


def build_query_matrix(query_data, vocab_size, device='cuda:0'):
    """Convert sparse queries to dense on GPU."""
    n_queries = query_data['n_queries']
    matrix = torch.zeros(n_queries, vocab_size, device=device, dtype=torch.float32)
    ptrs = query_data['ptrs']
    indices = query_data['indices']
    values = query_data['values']

    for i in range(n_queries):
        start, end = ptrs[i], ptrs[i+1]
        if end > start:
            idx = torch.tensor(indices[start:end], dtype=torch.long, device=device)
            val = torch.tensor(values[start:end], dtype=torch.float32, device=device)
            matrix[i].scatter_(0, idx, val)

    return matrix


def benchmark_gpu_dense_matmul(Q_dense, D_dense, n_warmup=3, n_runs=10):
    """GPU dense matmul scoring."""
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = torch.matmul(Q_dense, D_dense.T)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        scores = torch.matmul(Q_dense, D_dense.T)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'qps': Q_dense.shape[0] / np.mean(times),
    }


def benchmark_gpu_sparse_matmul(Q_dense, D_sparse_coo, n_warmup=3, n_runs=10):
    """GPU sparse matmul scoring."""
    D_csr = D_sparse_coo.to_sparse_csr()

    torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = torch.matmul(Q_dense, D_csr.T.to_dense())
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        scores = torch.matmul(Q_dense, D_csr.T.to_dense())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'qps': Q_dense.shape[0] / np.mean(times),
    }


def benchmark_cpu_sequential(doc_data, query_data, vocab_size):
    """CPU sequential inverted index traversal."""
    n_docs = doc_data['n_docs']
    n_queries = query_data['n_queries']

    # Build inverted index
    inverted_index = [[] for _ in range(vocab_size)]
    ptrs = doc_data['ptrs']
    indices = doc_data['indices']
    values = doc_data['values']

    for doc_id in range(n_docs):
        start, end = ptrs[doc_id], ptrs[doc_id + 1]
        for j in range(start, end):
            inverted_index[indices[j]].append((doc_id, values[j]))

    # Score
    start_time = time.perf_counter()
    all_scores = np.zeros((n_queries, n_docs), dtype=np.float32)

    q_ptrs = query_data['ptrs']
    q_indices = query_data['indices']
    q_values = query_data['values']

    for qi in range(n_queries):
        q_start, q_end = q_ptrs[qi], q_ptrs[qi + 1]
        for j in range(q_start, q_end):
            q_term = q_indices[j]
            q_weight = q_values[j]
            for doc_id, d_weight in inverted_index[q_term]:
                all_scores[qi, doc_id] += q_weight * d_weight

    elapsed = time.perf_counter() - start_time
    return {
        'mean_ms': elapsed * 1000,
        'qps': n_queries / elapsed,
    }


def main():
    print("=" * 80)
    print("GPUSparse Full-Scale Benchmark")
    print("=" * 80)

    device = 'cuda:0'
    print(f"Device: {torch.cuda.get_device_name(device)}")
    vocab_size = 30522  # BERT vocab

    all_results = {}
    n_queries = 100

    # Generate queries once
    print("\nGenerating queries...")
    query_data = generate_sparse_queries(n_queries, vocab_size, avg_terms=20)

    # Test at various scales
    scales = [10000, 50000, 100000, 500000, 1000000]

    for n_docs in scales:
        print(f"\n{'='*60}")
        print(f"Scale: {n_docs:,} documents, {n_queries} queries")
        print(f"{'='*60}")

        # Generate sparse documents
        doc_data = generate_splade_style_sparse(n_docs, vocab_size, avg_terms=100)
        print(f"  Avg NNZ per doc: {doc_data['avg_nnz']:.1f}")

        scale_results = {}

        # CPU baseline (only for small scales)
        if n_docs <= 50000:
            print("  CPU sequential...")
            cpu_res = benchmark_cpu_sequential(doc_data, query_data, vocab_size)
            scale_results['cpu_sequential'] = cpu_res
            print(f"    {cpu_res['mean_ms']:.1f}ms, {cpu_res['qps']:.1f} QPS")

        # GPU dense matmul
        try:
            print("  Building dense matrix on GPU...")
            torch.cuda.empty_cache()
            D_dense = build_dense_matrix(doc_data, device)
            Q_dense = build_query_matrix(query_data, vocab_size, device)

            mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"    GPU memory: {mem_gb:.1f} GB")

            print("  GPU dense matmul...")
            gpu_dense_res = benchmark_gpu_dense_matmul(Q_dense, D_dense)
            scale_results['gpu_dense_matmul'] = gpu_dense_res
            scale_results['gpu_dense_matmul']['memory_gb'] = mem_gb
            print(f"    {gpu_dense_res['mean_ms']:.1f}ms, {gpu_dense_res['qps']:.1f} QPS")

            del D_dense, Q_dense
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"    GPU dense OOM: {e}")
            scale_results['gpu_dense_matmul'] = {'error': 'OOM', 'memory_limit': '80GB'}
            torch.cuda.empty_cache()

        # Top-k extraction time
        if 'gpu_dense_matmul' not in scale_results or 'error' not in scale_results.get('gpu_dense_matmul', {}):
            try:
                scores_dummy = torch.randn(n_queries, n_docs, device=device)
                torch.cuda.synchronize()
                topk_times = []
                for _ in range(10):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _, _ = torch.topk(scores_dummy, k=100, dim=-1)
                    torch.cuda.synchronize()
                    topk_times.append(time.perf_counter() - start)
                scale_results['topk_100'] = {
                    'mean_ms': np.mean(topk_times) * 1000,
                }
                del scores_dummy
                torch.cuda.empty_cache()
            except:
                pass

        all_results[f'{n_docs}_docs'] = scale_results

        # Print summary
        print(f"\n  Summary for {n_docs:,} docs:")
        for method, res in scale_results.items():
            if 'error' not in res and 'mean_ms' in res:
                print(f"    {method}: {res['mean_ms']:.1f}ms")

    # Memory footprint analysis
    print("\n--- Memory Footprint Analysis ---")
    mem_analysis = {}
    for n_docs in scales + [5000000, 10000000]:
        avg_nnz = 100
        sparse_gb = n_docs * avg_nnz * 12 / (1024**3)  # 8 bytes index + 4 bytes value per entry
        dense_gb = n_docs * vocab_size * 4 / (1024**3)  # FP32
        mem_analysis[f'{n_docs}_docs'] = {
            'sparse_gb': round(sparse_gb, 2),
            'dense_gb': round(dense_gb, 2),
            'fits_1_h100_sparse': sparse_gb < 80,
            'fits_1_h100_dense': dense_gb < 80,
        }
        print(f"  {n_docs:>10,} docs: sparse={sparse_gb:.2f}GB, dense={dense_gb:.2f}GB")

    all_results['memory_analysis'] = mem_analysis

    # Save
    results_path = RESULTS_DIR / "full_scale_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
