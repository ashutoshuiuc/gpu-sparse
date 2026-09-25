"""
Fast benchmark for GPUSparse - optimized data generation and index building.
"""

import torch
import numpy as np
import time
import json
import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5,6")


def generate_fast_splade_data(num_docs, vocab_size=30522, avg_nnz=150, seed=42):
    """Generate sparse doc matrix directly as COO arrays."""
    rng = np.random.RandomState(seed)

    # Zipfian term probs
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, 1.5)
    probs /= probs.sum()

    # Pre-sample all terms at once
    total_nnz = num_docs * avg_nnz
    all_terms = rng.choice(vocab_size, size=total_nnz, replace=True, p=probs)
    all_scores = rng.uniform(0.1, 3.0, size=total_nnz).astype(np.float32)
    all_docs = np.repeat(np.arange(num_docs, dtype=np.int32), avg_nnz)

    return all_docs, all_terms.astype(np.int32), all_scores


def build_index_fast(doc_ids_arr, term_ids_arr, scores_arr, vocab_size, device):
    """Build GPU inverted index from COO arrays using numpy vectorization."""
    BLOCK = 32

    # Sort by term_id for grouping
    sort_idx = np.argsort(term_ids_arr)
    sorted_terms = term_ids_arr[sort_idx]
    sorted_docs = doc_ids_arr[sort_idx]
    sorted_scores = scores_arr[sort_idx]

    # Find boundaries for each term
    unique_terms, counts = np.unique(sorted_terms, return_counts=True)

    lengths = np.zeros(vocab_size, dtype=np.int32)
    lengths[unique_terms] = counts

    padded_lengths = ((lengths + BLOCK - 1) // BLOCK) * BLOCK
    offsets = np.zeros(vocab_size, dtype=np.int64)
    np.cumsum(padded_lengths[:-1], out=offsets[1:])

    total_padded = int(offsets[-1] + padded_lengths[-1])

    all_doc_ids = np.full(total_padded, -1, dtype=np.int32)
    all_scores_flat = np.zeros(total_padded, dtype=np.float32)

    # Fill posting lists
    src_offset = 0
    for i, term_id in enumerate(unique_terms):
        n = counts[i]
        off = offsets[term_id]
        all_doc_ids[off:off+n] = sorted_docs[src_offset:src_offset+n]
        all_scores_flat[off:off+n] = sorted_scores[src_offset:src_offset+n]
        src_offset += n

    # Compute max scores per term
    max_scores = np.zeros(vocab_size, dtype=np.float32)
    # Use split based on sorted data
    src_offset = 0
    for i, term_id in enumerate(unique_terms):
        n = counts[i]
        max_scores[term_id] = sorted_scores[src_offset:src_offset+n].max()
        src_offset += n

    # Transfer to GPU
    return {
        'doc_ids': torch.from_numpy(all_doc_ids).to(device),
        'scores': torch.from_numpy(all_scores_flat).to(device),
        'offsets': torch.from_numpy(offsets).to(device),
        'lengths': torch.from_numpy(lengths).to(device),
        'padded_lengths': torch.from_numpy(padded_lengths).to(device),
        'max_scores': torch.from_numpy(max_scores).to(device),
        'num_docs': int(doc_ids_arr.max() + 1),
        'vocab_size': vocab_size,
        'device': device,
    }


def make_queries(num_queries, vocab_size, max_terms=64, avg_terms=30, device=None, seed=123):
    """Synthetic query generator. NOTE: max_terms here is the width of the generated
    array, not a truncation of real queries, so it does not affect exactness. Real
    query preparation uses max_terms=128 (observed max nnz on MS MARCO dev is 107);
    see prepare_queries_from_meta in run_correctness_verification.py."""
    """Generate query batch."""
    rng = np.random.RandomState(seed)
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, 1.5)
    probs /= probs.sum()

    term_ids = np.full((num_queries, max_terms), -1, dtype=np.int32)
    term_scores = np.zeros((num_queries, max_terms), dtype=np.float32)

    for i in range(num_queries):
        n = max(5, min(max_terms, rng.poisson(avg_terms)))
        terms = rng.choice(vocab_size, size=n, replace=False, p=probs)
        scores = rng.uniform(0.1, 3.0, size=n).astype(np.float32)
        term_ids[i, :n] = terms
        term_scores[i, :n] = scores

    return (torch.from_numpy(term_ids).to(device),
            torch.from_numpy(term_scores).to(device))


def gpu_scatter_score(index, q_ids, q_scores, top_k=10):
    """GPU scatter-add scoring."""
    batch = q_ids.shape[0]
    max_qt = q_ids.shape[1]
    device = index['device']
    num_docs = index['num_docs']

    scores = torch.zeros(batch, num_docs, device=device, dtype=torch.float32)

    for t_pos in range(max_qt):
        terms = q_ids[:, t_pos]
        qs = q_scores[:, t_pos]

        for qi in range(batch):
            tid = terms[qi].item()
            if tid < 0:
                continue

            offset = index['offsets'][tid].item()
            length = index['lengths'][tid].item()
            if length == 0:
                continue

            pl_docs = index['doc_ids'][offset:offset+length]
            pl_scores = index['scores'][offset:offset+length]
            scores[qi].scatter_add_(0, pl_docs.long(), qs[qi] * pl_scores)

    return torch.topk(scores, k=min(top_k, num_docs), dim=1)


def gpu_wand_score(index, q_ids, q_scores, top_k=10, prune_ratio=0.5):
    """GPU WAND approximate scoring with term pruning."""
    batch = q_ids.shape[0]
    max_qt = q_ids.shape[1]
    device = index['device']
    num_docs = index['num_docs']

    # Compute upper bounds vectorized
    valid_mask = q_ids >= 0
    safe_ids = torch.where(valid_mask, q_ids, torch.zeros_like(q_ids))
    max_doc_scores = index['max_scores'][safe_ids.long()]
    upper_bounds = q_scores * max_doc_scores * valid_mask.float()

    # Per query: keep top (1-prune_ratio) terms
    n_keep = max(1, int(max_qt * (1.0 - prune_ratio)))
    _, top_term_indices = torch.topk(upper_bounds, k=n_keep, dim=1)

    # Create pruned query
    pruned_ids = torch.full_like(q_ids, -1)
    pruned_scores = torch.zeros_like(q_scores)
    for qi in range(batch):
        for j in range(n_keep):
            t_pos = top_term_indices[qi, j].item()
            pruned_ids[qi, j] = q_ids[qi, t_pos]
            pruned_scores[qi, j] = q_scores[qi, t_pos]

    return gpu_scatter_score(index, pruned_ids, pruned_scores, top_k)


def gpu_dense_matmul_baseline(doc_dense, q_dense, top_k=10):
    """Dense matmul baseline."""
    scores = torch.mm(q_dense, doc_dense.t())
    return torch.topk(scores, k=min(top_k, doc_dense.shape[0]), dim=1)


def cpu_sequential_baseline(all_docs, all_terms, all_scores, q_ids_np, q_scores_np, num_docs, top_k=10):
    """CPU sequential scoring baseline."""
    # Build simple inverted index as dict
    inv = {}
    for i in range(len(all_docs)):
        t = all_terms[i]
        if t not in inv:
            inv[t] = ([], [])
        inv[t][0].append(all_docs[i])
        inv[t][1].append(all_scores[i])

    results = []
    for qi in range(len(q_ids_np)):
        doc_scores = np.zeros(num_docs, dtype=np.float32)
        for t_pos in range(len(q_ids_np[qi])):
            tid = q_ids_np[qi][t_pos]
            if tid < 0:
                continue
            qs = q_scores_np[qi][t_pos]
            if tid in inv:
                for d, s in zip(inv[tid][0], inv[tid][1]):
                    doc_scores[d] += qs * s

        top_idx = np.argpartition(doc_scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(doc_scores[top_idx])[::-1]]
        results.append((doc_scores[top_idx], top_idx))

    return results


def benchmark_fn(fn, warmup=3, trials=10, sync_device=None):
    """Benchmark a function."""
    for _ in range(warmup):
        fn()
        if sync_device is not None:
            torch.cuda.synchronize(sync_device)

    times = []
    for _ in range(trials):
        if sync_device is not None:
            torch.cuda.synchronize(sync_device)
        t0 = time.perf_counter()
        fn()
        if sync_device is not None:
            torch.cuda.synchronize(sync_device)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
    }


def run_triton_benchmark(index, q_ids, q_scores, top_k, device):
    """Run Triton kernel benchmarks."""
    from src.triton_kernel import triton_fused_score, triton_wand_score
    from src.gpu_inverted_index import GPUInvertedIndex

    # Create a proper GPUInvertedIndex object
    gpu_idx = GPUInvertedIndex(
        doc_ids=index['doc_ids'],
        scores=index['scores'],
        offsets=index['offsets'],
        lengths=index['lengths'],
        padded_lengths=index['padded_lengths'],
        max_scores=index['max_scores'],
        num_docs=index['num_docs'],
        vocab_size=index['vocab_size'],
        device=index['device'],
    )

    results = {}

    print("  [Triton Fused]...", flush=True)
    try:
        r = benchmark_fn(lambda: triton_fused_score(gpu_idx, q_ids, q_scores, top_k),
                         warmup=3, trials=10, sync_device=device)
        results['triton_fused'] = r
        print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        results['triton_fused'] = {'error': str(e)}

    print("  [Triton WAND]...", flush=True)
    try:
        r = benchmark_fn(lambda: triton_wand_score(gpu_idx, q_ids, q_scores, top_k),
                         warmup=3, trials=10, sync_device=device)
        results['triton_wand'] = r
        print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        results['triton_wand'] = {'error': str(e)}

    return results


def main():
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"CUDA: {torch.version.cuda}", flush=True)

    all_results = {}
    vocab_size = 30522
    num_queries = 64
    top_k = 10

    for num_docs in [50_000, 100_000, 500_000, 1_000_000]:
        print(f"\n{'='*70}", flush=True)
        print(f"SCALE: {num_docs:,} documents", flush=True)
        print(f"{'='*70}", flush=True)

        results = {'num_docs': num_docs}

        # Generate data
        print("Generating data...", flush=True)
        t0 = time.time()
        doc_ids_arr, term_ids_arr, scores_arr = generate_fast_splade_data(
            num_docs, vocab_size, avg_nnz=150
        )
        print(f"  Generated {len(doc_ids_arr):,} postings in {time.time()-t0:.1f}s", flush=True)

        # Build GPU index
        print("Building GPU index...", flush=True)
        t0 = time.time()
        index = build_index_fast(doc_ids_arr, term_ids_arr, scores_arr, vocab_size, device)
        build_time = time.time() - t0
        print(f"  Built in {build_time:.1f}s", flush=True)

        total_mb = (index['doc_ids'].nelement() * 4 + index['scores'].nelement() * 4) / 1e6
        print(f"  Index size: {total_mb:.1f} MB on GPU", flush=True)
        results['index_mb'] = total_mb
        results['build_time_s'] = build_time

        # Queries
        q_ids, q_scores = make_queries(num_queries, vocab_size, device=device)

        # 1. GPU Scatter (ours)
        print("  [GPU Scatter]...", flush=True)
        r = benchmark_fn(lambda: gpu_scatter_score(index, q_ids, q_scores, top_k),
                         warmup=2, trials=5, sync_device=device)
        results['gpu_scatter'] = r
        print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")

        # 2. GPU WAND (ours)
        print("  [GPU WAND]...", flush=True)
        r = benchmark_fn(lambda: gpu_wand_score(index, q_ids, q_scores, top_k),
                         warmup=2, trials=5, sync_device=device)
        results['gpu_wand'] = r
        print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")

        # 3. Triton kernels
        triton_res = run_triton_benchmark(index, q_ids, q_scores, top_k, device)
        results.update(triton_res)

        # 4. Dense matmul (only up to 500K - memory)
        if num_docs <= 500_000:
            print("  [Dense MatMul]...", flush=True)
            try:
                # Build dense matrix
                doc_dense = torch.zeros(num_docs, vocab_size, device=device, dtype=torch.float32)
                # Fill from COO
                doc_dense[torch.from_numpy(doc_ids_arr).long().to(device),
                         torch.from_numpy(term_ids_arr).long().to(device)] = \
                    torch.from_numpy(scores_arr).to(device)

                q_dense = torch.zeros(num_queries, vocab_size, device=device, dtype=torch.float32)
                q_ids_cpu = q_ids.cpu().numpy()
                q_scores_cpu = q_scores.cpu().numpy()
                for qi in range(num_queries):
                    for j in range(64):
                        if q_ids_cpu[qi, j] >= 0:
                            q_dense[qi, q_ids_cpu[qi, j]] = q_scores_cpu[qi, j]

                r = benchmark_fn(lambda: gpu_dense_matmul_baseline(doc_dense, q_dense, top_k),
                                 warmup=3, trials=10, sync_device=device)
                results['gpu_dense_matmul'] = r
                print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
                del doc_dense, q_dense
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    FAILED: {e}")
                results['gpu_dense_matmul'] = {'error': str(e)}
        else:
            print("  [Dense MatMul] SKIPPED (memory)")
            results['gpu_dense_matmul'] = {'skipped': True}

        # 5. Sparse matmul
        if num_docs <= 500_000:
            print("  [Sparse MatMul]...", flush=True)
            try:
                indices = torch.tensor(
                    [doc_ids_arr.astype(np.int64), term_ids_arr.astype(np.int64)],
                    dtype=torch.int64
                )
                values = torch.from_numpy(scores_arr)
                doc_sparse = torch.sparse_coo_tensor(
                    indices, values, (num_docs, vocab_size)
                ).to_sparse_csr().to(device)

                q_rows, q_cols, q_vals = [], [], []
                q_ids_cpu = q_ids.cpu().numpy()
                q_scores_cpu = q_scores.cpu().numpy()
                for qi in range(num_queries):
                    for j in range(64):
                        if q_ids_cpu[qi, j] >= 0:
                            q_rows.append(qi)
                            q_cols.append(int(q_ids_cpu[qi, j]))
                            q_vals.append(float(q_scores_cpu[qi, j]))

                q_indices = torch.tensor([q_rows, q_cols], dtype=torch.int64)
                q_values = torch.tensor(q_vals, dtype=torch.float32)
                q_sparse = torch.sparse_coo_tensor(
                    q_indices, q_values, (num_queries, vocab_size)
                ).to_sparse_csr().to(device)

                r = benchmark_fn(
                    lambda: torch.topk(
                        torch.sparse.mm(q_sparse, doc_sparse.t().to_dense()).to_dense()
                        if torch.sparse.mm(q_sparse, doc_sparse.t().to_dense()).is_sparse
                        else torch.sparse.mm(q_sparse, doc_sparse.t().to_dense()),
                        k=top_k, dim=1
                    ),
                    warmup=2, trials=5, sync_device=device
                )
                results['gpu_sparse_matmul'] = r
                print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
                del doc_sparse, q_sparse
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    FAILED: {e}")
                results['gpu_sparse_matmul'] = {'error': str(e)}
        else:
            print("  [Sparse MatMul] SKIPPED")
            results['gpu_sparse_matmul'] = {'skipped': True}

        # 6. CPU sequential (only small scale)
        if num_docs <= 50_000:
            print("  [CPU Sequential]...", flush=True)
            q_ids_cpu = q_ids.cpu().numpy()
            q_scores_cpu = q_scores.cpu().numpy()
            r = benchmark_fn(
                lambda: cpu_sequential_baseline(
                    doc_ids_arr, term_ids_arr, scores_arr,
                    q_ids_cpu, q_scores_cpu, num_docs, top_k
                ),
                warmup=1, trials=3, sync_device=None
            )
            results['cpu_sequential'] = r
            print(f"    {r['mean_ms']:.2f} +/- {r['std_ms']:.2f} ms")
        else:
            print("  [CPU Sequential] SKIPPED")
            results['cpu_sequential'] = {'skipped': True}

        all_results[str(num_docs)] = results

        # Cleanup
        del index
        gc.collect()
        torch.cuda.empty_cache()

    # GPU memory profile
    print(f"\n{'='*70}", flush=True)
    print("GPU MEMORY PROFILE", flush=True)
    print(f"{'='*70}", flush=True)
    for num_docs in [100_000, 500_000, 1_000_000]:
        torch.cuda.reset_peak_memory_stats(device)
        doc_ids_arr, term_ids_arr, scores_arr = generate_fast_splade_data(num_docs, vocab_size)
        index = build_index_fast(doc_ids_arr, term_ids_arr, scores_arr, vocab_size, device)
        q_ids, q_scores = make_queries(64, vocab_size, device=device)

        mem_before = torch.cuda.memory_allocated(device) / 1e6
        gpu_scatter_score(index, q_ids, q_scores, top_k)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1e6

        print(f"  {num_docs:>10,} docs: allocated={mem_before:.0f} MB, peak={peak:.0f} MB")

        del index
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tracker", "benchmark_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<25s} ", end="")
    for scale in sorted(all_results.keys(), key=int):
        print(f"{'%s docs' % f'{int(scale):,}':>18s} ", end="")
    print()
    print("-" * 100)

    for method in ['gpu_scatter', 'gpu_wand', 'triton_fused', 'triton_wand',
                    'gpu_dense_matmul', 'gpu_sparse_matmul', 'cpu_sequential']:
        print(f"{method:<25s} ", end="")
        for scale in sorted(all_results.keys(), key=int):
            r = all_results[scale].get(method, {})
            if 'mean_ms' in r:
                print(f"{r['mean_ms']:>15.2f} ms ", end="")
            elif 'skipped' in r:
                print(f"{'SKIP':>18s} ", end="")
            elif 'error' in r:
                print(f"{'ERR':>18s} ", end="")
            else:
                print(f"{'--':>18s} ", end="")
        print()


if __name__ == "__main__":
    main()
