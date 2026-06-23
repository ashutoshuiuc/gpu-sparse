"""
Comprehensive Benchmark Suite for GPUSparse.

Benchmarks:
1. GPU Scatter Scoring (ours)
2. GPU WAND Approximation (ours)
3. Fused Triton Kernel (ours)
4. Triton WAND Kernel (ours)
5. GPU Dense Matmul baseline
6. GPU Sparse Matmul baseline (torch.sparse)
7. CPU Sequential baseline

Scale tests: 100K, 1M, 10M documents
"""

import torch
import numpy as np
import time
import json
import os
import sys
import gc

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpu_inverted_index import (
    build_gpu_inverted_index,
    generate_synthetic_splade_data,
    index_memory_stats,
)
from src.gpu_scoring import (
    gpu_scatter_score_vectorized,
    gpu_wand_score,
    gpu_dense_matmul_score,
    gpu_sparse_matmul_score,
    cpu_sequential_score,
)
from src.triton_kernel import triton_fused_score, triton_wand_score


def make_query_batch(
    num_queries: int,
    vocab_size: int,
    avg_terms: int = 30,
    max_terms: int = 64,
    device: torch.device = torch.device("cuda:0"),
    seed: int = 123,
):
    """Generate a batch of synthetic queries."""
    rng = np.random.RandomState(seed)
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, 1.5)
    probs /= probs.sum()

    term_ids = np.full((num_queries, max_terms), -1, dtype=np.int32)
    term_scores = np.zeros((num_queries, max_terms), dtype=np.float32)

    query_ids_list = []
    query_scores_list = []

    for i in range(num_queries):
        n = max(5, min(max_terms, rng.poisson(avg_terms)))
        terms = rng.choice(vocab_size, size=n, replace=False, p=probs)
        scores = rng.uniform(0.1, 3.0, size=n).astype(np.float32)
        term_ids[i, :n] = terms
        term_scores[i, :n] = scores
        query_ids_list.append(terms.tolist())
        query_scores_list.append(scores.tolist())

    q_ids = torch.from_numpy(term_ids).to(device)
    q_scores = torch.from_numpy(term_scores).to(device)

    return q_ids, q_scores, query_ids_list, query_scores_list


def build_dense_matrix(doc_term_ids, doc_term_scores, num_docs, vocab_size, device):
    """Build dense document matrix on GPU."""
    mat = torch.zeros(num_docs, vocab_size, dtype=torch.float32)
    for i in range(num_docs):
        for tid, score in zip(doc_term_ids[i], doc_term_scores[i]):
            mat[i, tid] = score
    return mat.to(device)


def build_sparse_csr(doc_term_ids, doc_term_scores, num_docs, vocab_size, device):
    """Build sparse CSR matrix on GPU."""
    rows, cols, vals = [], [], []
    for i in range(num_docs):
        for tid, score in zip(doc_term_ids[i], doc_term_scores[i]):
            rows.append(i)
            cols.append(tid)
            vals.append(score)

    indices = torch.tensor([rows, cols], dtype=torch.int64)
    values = torch.tensor(vals, dtype=torch.float32)
    sparse = torch.sparse_coo_tensor(indices, values, (num_docs, vocab_size))
    return sparse.to_sparse_csr().to(device)


def benchmark_method(fn, warmup=3, trials=10, sync=True):
    """Benchmark a callable, return mean and std latency in ms."""
    device = torch.device("cuda:0")

    # Warmup
    for _ in range(warmup):
        fn()
        if sync:
            torch.cuda.synchronize(device)

    latencies = []
    for _ in range(trials):
        if sync:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "trials": trials,
    }


def run_scale_benchmark(
    num_docs: int,
    vocab_size: int = 30522,
    num_queries: int = 64,
    top_k: int = 10,
    avg_terms_per_doc: int = 150,
    device: torch.device = torch.device("cuda:0"),
    run_cpu: bool = True,
    run_dense: bool = True,
    run_sparse_mm: bool = True,
):
    """Run full benchmark suite at a given document scale."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {num_docs:,} docs, {vocab_size:,} vocab, {num_queries} queries")
    print(f"{'='*70}")

    # Generate data
    print("Generating synthetic SPLADE data...")
    t0 = time.time()
    doc_ids, doc_scores = generate_synthetic_splade_data(
        num_docs, vocab_size, avg_terms_per_doc
    )
    print(f"  Data generation: {time.time()-t0:.1f}s")

    # Build GPU inverted index
    print("Building GPU inverted index...")
    t0 = time.time()
    index = build_gpu_inverted_index(doc_ids, doc_scores, vocab_size, device)
    print(f"  Index build: {time.time()-t0:.1f}s")

    mem = index_memory_stats(index)
    print(f"  Index memory: {mem['total_MB']:.1f} MB "
          f"({mem['num_postings']:,} postings, {mem['padding_overhead']:.1%} padding overhead)")

    # Generate queries
    q_ids, q_scores, q_ids_list, q_scores_list = make_query_batch(
        num_queries, vocab_size, device=device
    )

    results = {"num_docs": num_docs, "vocab_size": vocab_size, "num_queries": num_queries}

    # 1. GPU Scatter Score (Ours)
    print("\n[1] GPU Scatter Score (vectorized)...")
    try:
        r = benchmark_method(lambda: gpu_scatter_score_vectorized(index, q_ids, q_scores, top_k))
        results["gpu_scatter"] = r
        print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["gpu_scatter"] = {"error": str(e)}

    # 2. GPU WAND Approximation (Ours)
    print("[2] GPU WAND Approximation...")
    try:
        r = benchmark_method(lambda: gpu_wand_score(index, q_ids, q_scores, top_k))
        results["gpu_wand"] = r
        print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["gpu_wand"] = {"error": str(e)}

    # 3. Fused Triton Kernel (Ours)
    print("[3] Fused Triton Scatter Kernel...")
    try:
        r = benchmark_method(lambda: triton_fused_score(index, q_ids, q_scores, top_k))
        results["triton_fused"] = r
        print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["triton_fused"] = {"error": str(e)}

    # 4. Triton WAND Kernel (Ours)
    print("[4] Triton WAND Kernel...")
    try:
        r = benchmark_method(lambda: triton_wand_score(index, q_ids, q_scores, top_k))
        results["triton_wand"] = r
        print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["triton_wand"] = {"error": str(e)}

    # 5. Dense matmul baseline
    if run_dense and num_docs <= 1_000_000:
        print("[5] GPU Dense Matmul...")
        try:
            dense_mat = build_dense_matrix(doc_ids, doc_scores, num_docs, vocab_size, device)
            q_dense = torch.zeros(num_queries, vocab_size, device=device)
            for i in range(num_queries):
                for tid, sc in zip(q_ids_list[i], q_scores_list[i]):
                    q_dense[i, tid] = sc

            r = benchmark_method(lambda: gpu_dense_matmul_score(dense_mat, q_dense, top_k))
            results["gpu_dense_matmul"] = r
            print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
            del dense_mat, q_dense
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    FAILED: {e}")
            results["gpu_dense_matmul"] = {"error": str(e)}
    else:
        print("[5] GPU Dense Matmul... SKIPPED (too large)")
        results["gpu_dense_matmul"] = {"skipped": True}

    # 6. Sparse matmul baseline
    if run_sparse_mm and num_docs <= 1_000_000:
        print("[6] GPU Sparse Matmul (torch.sparse)...")
        try:
            sparse_docs = build_sparse_csr(doc_ids, doc_scores, num_docs, vocab_size, device)
            sparse_queries = build_sparse_csr(
                q_ids_list, q_scores_list, num_queries, vocab_size, device
            )
            r = benchmark_method(
                lambda: gpu_sparse_matmul_score(sparse_docs, sparse_queries, top_k)
            )
            results["gpu_sparse_matmul"] = r
            print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
            del sparse_docs, sparse_queries
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    FAILED: {e}")
            results["gpu_sparse_matmul"] = {"error": str(e)}
    else:
        print("[6] GPU Sparse Matmul... SKIPPED (too large)")
        results["gpu_sparse_matmul"] = {"skipped": True}

    # 7. CPU Sequential baseline
    if run_cpu and num_docs <= 100_000:
        print("[7] CPU Sequential...")
        try:
            r = benchmark_method(
                lambda: cpu_sequential_score(
                    doc_ids, doc_scores, q_ids_list, q_scores_list, num_docs, top_k
                ),
                warmup=1, trials=3, sync=False,
            )
            results["cpu_sequential"] = r
            print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {e}")
            results["cpu_sequential"] = {"error": str(e)}
    else:
        print("[7] CPU Sequential... SKIPPED (too large)")
        results["cpu_sequential"] = {"skipped": True}

    # Memory stats
    results["index_memory"] = mem

    # Cleanup
    del index
    gc.collect()
    torch.cuda.empty_cache()

    return results


def correctness_check(device):
    """Verify all methods return the same top-k results."""
    print("\n" + "="*70)
    print("CORRECTNESS CHECK")
    print("="*70)

    num_docs = 1000
    vocab_size = 500
    num_queries = 4
    top_k = 5

    doc_ids, doc_scores = generate_synthetic_splade_data(
        num_docs, vocab_size, avg_terms_per_doc=50, seed=42
    )
    index = build_gpu_inverted_index(doc_ids, doc_scores, vocab_size, device)
    q_ids, q_scores, q_ids_list, q_scores_list = make_query_batch(
        num_queries, vocab_size, avg_terms=15, max_terms=32, device=device, seed=99
    )

    # Reference: dense matmul
    dense_mat = build_dense_matrix(doc_ids, doc_scores, num_docs, vocab_size, device)
    q_dense = torch.zeros(num_queries, vocab_size, device=device)
    for i in range(num_queries):
        for tid, sc in zip(q_ids_list[i], q_scores_list[i]):
            q_dense[i, tid] = sc

    ref_scores, ref_ids = gpu_dense_matmul_score(dense_mat, q_dense, top_k)

    methods = [
        ("GPU Scatter", lambda: gpu_scatter_score_vectorized(index, q_ids, q_scores, top_k)),
        ("GPU WAND (50% prune)", lambda: gpu_wand_score(index, q_ids, q_scores, top_k, prune_ratio=0.5)),
        ("Triton Fused", lambda: triton_fused_score(index, q_ids, q_scores, top_k)),
        ("Triton WAND", lambda: triton_wand_score(index, q_ids, q_scores, top_k)),
    ]

    for name, fn in methods:
        scores, ids = fn()
        # Check if top-1 matches
        top1_match = (ids[:, 0] == ref_ids[:, 0]).all().item()
        # Check score correlation
        score_diff = (scores[:, 0] - ref_scores[:, 0]).abs().mean().item()
        # Top-5 overlap
        overlap = 0
        for q in range(num_queries):
            ref_set = set(ref_ids[q].cpu().tolist())
            test_set = set(ids[q].cpu().tolist())
            overlap += len(ref_set & test_set) / len(ref_set)
        overlap /= num_queries

        status = "PASS" if (top1_match and overlap > 0.6) else "WARN"
        print(f"  [{status}] {name}: top1_match={top1_match}, "
              f"top5_overlap={overlap:.2f}, score_diff={score_diff:.4f}")

    print("Correctness check complete.\n")


def profile_gpu_utilization(device):
    """Profile GPU memory and compute utilization."""
    print("\n" + "="*70)
    print("GPU UTILIZATION PROFILE")
    print("="*70)

    for num_docs in [50_000, 100_000, 500_000]:
        doc_ids, doc_scores = generate_synthetic_splade_data(num_docs, 30522, 150)
        index = build_gpu_inverted_index(doc_ids, doc_scores, 30522, device)
        q_ids, q_scores, _, _ = make_query_batch(64, 30522, device=device)

        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device) / 1e6

        # Run scoring
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(10):
            triton_fused_score(index, q_ids, q_scores, 10)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
        avg_ms = (t1 - t0) / 10 * 1000

        print(f"  {num_docs:>8,} docs: {avg_ms:.2f} ms/batch, "
              f"mem_alloc={mem_before:.0f} MB, peak={peak_mem:.0f} MB")

        del index
        gc.collect()
        torch.cuda.empty_cache()


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5,6")
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA: {torch.version.cuda}")

    all_results = {}

    # Correctness check
    correctness_check(device)

    # Scale benchmarks
    for num_docs in [100_000, 1_000_000]:
        try:
            results = run_scale_benchmark(
                num_docs=num_docs,
                device=device,
                run_cpu=(num_docs <= 100_000),
                run_dense=(num_docs <= 1_000_000),
                run_sparse_mm=(num_docs <= 1_000_000),
            )
            all_results[f"{num_docs}"] = results
        except Exception as e:
            print(f"FAILED for {num_docs}: {e}")
            import traceback
            traceback.print_exc()

    # GPU utilization profile
    profile_gpu_utilization(device)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tracker", "benchmark_results.json"
    )
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for scale, res in all_results.items():
        print(f"\n--- {int(scale):,} documents ---")
        for method in ["gpu_scatter", "gpu_wand", "triton_fused", "triton_wand",
                        "gpu_dense_matmul", "gpu_sparse_matmul", "cpu_sequential"]:
            if method in res and "mean_ms" in res[method]:
                print(f"  {method:25s}: {res[method]['mean_ms']:8.2f} ms")
            elif method in res and "skipped" in res[method]:
                print(f"  {method:25s}: SKIPPED")
            elif method in res:
                print(f"  {method:25s}: ERROR")


if __name__ == "__main__":
    main()
