"""
Comprehensive benchmark for GPUSparse paper - covers all experiment dimensions:
1. Scale: 100K, 500K, 1M, 5M documents
2. Sparsity: avg 10, 50, 100, 200, 500 terms/doc
3. Vocabulary size: 10K, 30K, 50K, 100K
4. Batch sizes: 1, 8, 32, 64, 128, 256, 512, 1000 queries
5. Method comparison: dense matmul, sparse matmul, Triton fused, Triton WAND
6. Memory usage analysis at each scale
7. Top-k benchmark: k = 1, 10, 100, 1000
8. Multi-GPU: shard index across 2 GPUs
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

from src.triton_kernel import triton_fused_score, triton_wand_score
from src.gpu_inverted_index import GPUInvertedIndex


def generate_data(num_docs, vocab_size=30522, avg_nnz=150, seed=42):
    """Generate sparse doc matrix as COO arrays."""
    rng = np.random.RandomState(seed)
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, 1.5)
    probs /= probs.sum()
    total_nnz = num_docs * avg_nnz
    all_terms = rng.choice(vocab_size, size=total_nnz, replace=True, p=probs).astype(np.int32)
    all_scores = rng.uniform(0.1, 3.0, size=total_nnz).astype(np.float32)
    all_docs = np.repeat(np.arange(num_docs, dtype=np.int32), avg_nnz)
    return all_docs, all_terms, all_scores


def build_index(doc_ids_arr, term_ids_arr, scores_arr, vocab_size, device):
    """Build GPU inverted index from COO arrays."""
    BLOCK = 32
    sort_idx = np.argsort(term_ids_arr)
    sorted_terms = term_ids_arr[sort_idx]
    sorted_docs = doc_ids_arr[sort_idx]
    sorted_scores = scores_arr[sort_idx]
    unique_terms, counts = np.unique(sorted_terms, return_counts=True)
    lengths = np.zeros(vocab_size, dtype=np.int32)
    lengths[unique_terms] = counts
    padded_lengths = ((lengths + BLOCK - 1) // BLOCK) * BLOCK
    offsets = np.zeros(vocab_size, dtype=np.int64)
    np.cumsum(padded_lengths[:-1], out=offsets[1:])
    total_padded = int(offsets[-1] + padded_lengths[-1])
    all_doc_ids = np.full(total_padded, -1, dtype=np.int32)
    all_scores_flat = np.zeros(total_padded, dtype=np.float32)
    max_scores = np.zeros(vocab_size, dtype=np.float32)
    src_offset = 0
    for i, term_id in enumerate(unique_terms):
        n = counts[i]
        off = offsets[term_id]
        all_doc_ids[off:off+n] = sorted_docs[src_offset:src_offset+n]
        all_scores_flat[off:off+n] = sorted_scores[src_offset:src_offset+n]
        max_scores[term_id] = sorted_scores[src_offset:src_offset+n].max()
        src_offset += n
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


def dict_to_index(d):
    """Convert dict to GPUInvertedIndex."""
    return GPUInvertedIndex(
        doc_ids=d['doc_ids'], scores=d['scores'],
        offsets=d['offsets'], lengths=d['lengths'],
        padded_lengths=d['padded_lengths'], max_scores=d['max_scores'],
        num_docs=d['num_docs'], vocab_size=d['vocab_size'], device=d['device'],
    )


def make_queries(num_queries, vocab_size, max_terms=64, avg_terms=30, device=None, seed=123):
    """Synthetic query generator; max_terms is the generated array width, not a
    truncation of real queries, so it does not affect exactness. Real query
    preparation uses max_terms=128."""
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


def bench(fn, warmup=3, trials=10, sync_device=None):
    """Benchmark a function, return dict with mean/std/min ms."""
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
    return {'mean_ms': float(np.mean(times)), 'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times))}


def safe_run(label, fn, **kwargs):
    """Run benchmark, catching OOM and other errors."""
    print(f"  [{label}]...", flush=True)
    try:
        r = bench(fn, **kwargs)
        print(f"    {r['mean_ms']:.3f} +/- {r['std_ms']:.3f} ms")
        return r
    except torch.cuda.OutOfMemoryError as e:
        print(f"    OOM: {e}")
        torch.cuda.empty_cache()
        return {'error': 'OOM'}
    except Exception as e:
        print(f"    FAILED: {e}")
        torch.cuda.empty_cache()
        return {'error': str(e)[:200]}


# =====================================================================
# EXPERIMENT 1: Scale (100K, 500K, 1M, 5M) with Triton fused
# =====================================================================
def exp_scale(device):
    print("\n" + "="*70)
    print("EXPERIMENT 1: SCALING WITH DOCUMENT COUNT")
    print("="*70)
    results = {}
    vocab_size = 30522
    for num_docs in [100_000, 500_000, 1_000_000, 5_000_000]:
        label = f"{num_docs//1000}K"
        print(f"\n--- {num_docs:,} documents ---", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        doc_ids, term_ids, scores = generate_data(num_docs, vocab_size, avg_nnz=150)
        gen_time = time.time() - t0
        print(f"  Data generated in {gen_time:.1f}s", flush=True)

        t0 = time.time()
        idx = build_index(doc_ids, term_ids, scores, vocab_size, device)
        build_time = time.time() - t0
        idx_obj = dict_to_index(idx)
        total_mb = (idx['doc_ids'].nelement() * 4 + idx['scores'].nelement() * 4) / 1e6
        print(f"  Index: {total_mb:.0f} MB, built in {build_time:.1f}s", flush=True)

        q_ids, q_scores = make_queries(64, vocab_size, device=device)

        mem_before = torch.cuda.memory_allocated(device) / 1e9
        r = safe_run("Triton Fused", lambda: triton_fused_score(idx_obj, q_ids, q_scores, 10),
                      warmup=3, trials=10, sync_device=device)
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

        results[label] = {
            'num_docs': num_docs, 'index_mb': total_mb, 'build_time_s': build_time,
            'triton_fused': r, 'mem_alloc_gb': round(mem_before, 2),
            'mem_peak_gb': round(peak_mem, 2),
            'per_query_us': round(r['mean_ms'] * 1000 / 64, 1) if 'mean_ms' in r else None,
            'throughput_qps': round(64 / (r['mean_ms'] / 1000), 0) if 'mean_ms' in r else None,
        }

        # Also run Triton WAND
        r2 = safe_run("Triton WAND", lambda: triton_wand_score(idx_obj, q_ids, q_scores, 10),
                       warmup=3, trials=10, sync_device=device)
        results[label]['triton_wand'] = r2

        # Sparse matmul only at smaller scales
        if num_docs <= 100_000:
            try:
                indices = torch.tensor([doc_ids.astype(np.int64), term_ids.astype(np.int64)], dtype=torch.int64)
                values = torch.from_numpy(scores)
                doc_sparse = torch.sparse_coo_tensor(indices, values, (num_docs, vocab_size)).to_sparse_csr().to(device)
                q_ids_cpu = q_ids.cpu().numpy(); q_scores_cpu = q_scores.cpu().numpy()
                q_rows, q_cols, q_vals = [], [], []
                for qi in range(64):
                    for j in range(64):
                        if q_ids_cpu[qi, j] >= 0:
                            q_rows.append(qi); q_cols.append(int(q_ids_cpu[qi, j])); q_vals.append(float(q_scores_cpu[qi, j]))
                q_sp = torch.sparse_coo_tensor(torch.tensor([q_rows, q_cols], dtype=torch.int64),
                                               torch.tensor(q_vals, dtype=torch.float32),
                                               (64, vocab_size)).to_sparse_csr().to(device)

                def _sparse_mm():
                    s = torch.sparse.mm(q_sp, doc_sparse.t().to_dense())
                    if s.is_sparse: s = s.to_dense()
                    return torch.topk(s, k=10, dim=1)

                r3 = safe_run("Sparse MatMul", _sparse_mm, warmup=2, trials=5, sync_device=device)
                results[label]['sparse_matmul'] = r3
                del doc_sparse, q_sp
                torch.cuda.empty_cache()
            except Exception as e:
                results[label]['sparse_matmul'] = {'error': str(e)[:200]}

        del idx, idx_obj, doc_ids, term_ids, scores
        gc.collect(); torch.cuda.empty_cache()

    return results


# =====================================================================
# EXPERIMENT 2: Vary sparsity (terms per doc)
# =====================================================================
def exp_sparsity(device):
    print("\n" + "="*70)
    print("EXPERIMENT 2: VARYING SPARSITY (TERMS PER DOCUMENT)")
    print("="*70)
    results = {}
    num_docs = 500_000
    vocab_size = 30522
    for avg_nnz in [10, 50, 100, 200, 500]:
        print(f"\n--- avg_nnz={avg_nnz} ---", flush=True)
        gc.collect(); torch.cuda.empty_cache()

        doc_ids, term_ids, scores = generate_data(num_docs, vocab_size, avg_nnz=avg_nnz)
        t0 = time.time()
        idx = build_index(doc_ids, term_ids, scores, vocab_size, device)
        build_time = time.time() - t0
        idx_obj = dict_to_index(idx)
        total_mb = (idx['doc_ids'].nelement() * 4 + idx['scores'].nelement() * 4) / 1e6
        print(f"  Index: {total_mb:.0f} MB, built in {build_time:.1f}s", flush=True)

        q_ids, q_scores = make_queries(64, vocab_size, device=device)
        r = safe_run("Triton Fused", lambda: triton_fused_score(idx_obj, q_ids, q_scores, 10),
                      warmup=3, trials=10, sync_device=device)

        results[str(avg_nnz)] = {
            'avg_nnz': avg_nnz, 'index_mb': total_mb, 'build_time_s': build_time,
            'triton_fused': r,
        }
        del idx, idx_obj, doc_ids, term_ids, scores
        gc.collect(); torch.cuda.empty_cache()

    return results


# =====================================================================
# EXPERIMENT 3: Vary vocabulary size
# =====================================================================
def exp_vocab(device):
    print("\n" + "="*70)
    print("EXPERIMENT 3: VARYING VOCABULARY SIZE")
    print("="*70)
    results = {}
    num_docs = 500_000
    for vocab_size in [10_000, 30_000, 50_000, 100_000]:
        print(f"\n--- vocab_size={vocab_size:,} ---", flush=True)
        gc.collect(); torch.cuda.empty_cache()

        doc_ids, term_ids, scores = generate_data(num_docs, vocab_size, avg_nnz=150)
        t0 = time.time()
        idx = build_index(doc_ids, term_ids, scores, vocab_size, device)
        build_time = time.time() - t0
        idx_obj = dict_to_index(idx)
        total_mb = (idx['doc_ids'].nelement() * 4 + idx['scores'].nelement() * 4) / 1e6
        print(f"  Index: {total_mb:.0f} MB, built in {build_time:.1f}s", flush=True)

        q_ids, q_scores = make_queries(64, vocab_size, device=device)
        r = safe_run("Triton Fused", lambda: triton_fused_score(idx_obj, q_ids, q_scores, 10),
                      warmup=3, trials=10, sync_device=device)

        results[str(vocab_size)] = {
            'vocab_size': vocab_size, 'index_mb': total_mb, 'build_time_s': build_time,
            'triton_fused': r,
        }
        del idx, idx_obj, doc_ids, term_ids, scores
        gc.collect(); torch.cuda.empty_cache()

    return results


# =====================================================================
# EXPERIMENT 4: Vary batch size
# =====================================================================
def exp_batch(device):
    print("\n" + "="*70)
    print("EXPERIMENT 4: VARYING BATCH SIZE")
    print("="*70)
    results = {}
    num_docs = 500_000
    vocab_size = 30522

    doc_ids, term_ids, scores = generate_data(num_docs, vocab_size, avg_nnz=150)
    idx = build_index(doc_ids, term_ids, scores, vocab_size, device)
    idx_obj = dict_to_index(idx)

    for batch_size in [1, 8, 32, 64, 128, 256, 512, 1000]:
        print(f"\n--- batch_size={batch_size} ---", flush=True)
        gc.collect(); torch.cuda.empty_cache()

        q_ids, q_scores = make_queries(batch_size, vocab_size, device=device)

        r = safe_run("Triton Fused", lambda: triton_fused_score(idx_obj, q_ids, q_scores, 10),
                      warmup=3, trials=10, sync_device=device)

        per_query = r['mean_ms'] / batch_size * 1000 if 'mean_ms' in r else None  # us
        throughput = batch_size / (r['mean_ms'] / 1000) if 'mean_ms' in r else None  # qps

        results[str(batch_size)] = {
            'batch_size': batch_size, 'triton_fused': r,
            'per_query_us': round(per_query, 1) if per_query else None,
            'throughput_qps': round(throughput, 0) if throughput else None,
        }

    del idx, idx_obj, doc_ids, term_ids, scores
    gc.collect(); torch.cuda.empty_cache()
    return results


# =====================================================================
# EXPERIMENT 5: Top-k extraction benchmark
# =====================================================================
def exp_topk(device):
    print("\n" + "="*70)
    print("EXPERIMENT 5: TOP-K EXTRACTION")
    print("="*70)
    results = {}
    for num_docs in [100_000, 1_000_000, 5_000_000]:
        for k in [1, 10, 100, 1000]:
            label = f"N={num_docs//1000}K_k={k}"
            print(f"\n--- {label} ---", flush=True)
            gc.collect(); torch.cuda.empty_cache()

            scores = torch.randn(64, num_docs, device=device, dtype=torch.float32)
            r = safe_run(f"topk(k={k})", lambda: torch.topk(scores, k=min(k, num_docs), dim=1),
                          warmup=5, trials=20, sync_device=device)
            results[label] = {'num_docs': num_docs, 'k': k, 'topk': r}
            del scores

    gc.collect(); torch.cuda.empty_cache()
    return results


# =====================================================================
# EXPERIMENT 6: Memory footprint analysis
# =====================================================================
def exp_memory(device):
    print("\n" + "="*70)
    print("EXPERIMENT 6: MEMORY FOOTPRINT ANALYSIS")
    print("="*70)
    results = {}
    vocab_size = 30522
    for num_docs in [100_000, 500_000, 1_000_000, 5_000_000]:
        label = f"{num_docs//1000}K"
        print(f"\n--- {num_docs:,} documents ---", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        doc_ids, term_ids, scores = generate_data(num_docs, vocab_size, avg_nnz=150)
        idx = build_index(doc_ids, term_ids, scores, vocab_size, device)
        idx_obj = dict_to_index(idx)

        index_mb = (idx['doc_ids'].nelement() * 4 + idx['scores'].nelement() * 4) / 1e6
        meta_mb = (idx['offsets'].nelement() * 8 + idx['lengths'].nelement() * 4 +
                   idx['padded_lengths'].nelement() * 4 + idx['max_scores'].nelement() * 4) / 1e6

        # Measure scoring memory overhead
        q_ids, q_scores = make_queries(64, vocab_size, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device) / 1e6

        triton_fused_score(idx_obj, q_ids, q_scores, 10)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1e6
        scoring_overhead = peak - mem_before

        # Score buffer theoretical size
        score_buffer_mb = 64 * num_docs * 4 / 1e6

        results[label] = {
            'num_docs': num_docs, 'index_mb': round(index_mb, 1),
            'metadata_mb': round(meta_mb, 2),
            'scoring_overhead_mb': round(scoring_overhead, 1),
            'score_buffer_theoretical_mb': round(score_buffer_mb, 1),
            'total_peak_mb': round(peak, 1),
            'pct_80gb_used': round(peak / (80 * 1024) * 100, 1),
        }
        print(f"  Index: {index_mb:.0f} MB, Score buffer: {score_buffer_mb:.0f} MB, "
              f"Peak: {peak:.0f} MB ({peak/(80*1024)*100:.1f}% of 80GB)", flush=True)

        del idx, idx_obj, doc_ids, term_ids, scores
        gc.collect(); torch.cuda.empty_cache()

    return results


# =====================================================================
# EXPERIMENT 7: Multi-GPU sharding (2 GPUs)
# =====================================================================
def exp_multi_gpu():
    print("\n" + "="*70)
    print("EXPERIMENT 7: MULTI-GPU SHARDING")
    print("="*70)

    n_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {n_gpus}")
    if n_gpus < 2:
        print("  SKIPPED: need at least 2 GPUs")
        return {'skipped': True, 'reason': f'only {n_gpus} GPU(s)'}

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    results = {}
    vocab_size = 30522
    num_docs = 2_000_000  # 2M total, 1M per GPU

    print(f"\n--- {num_docs:,} documents across 2 GPUs ---", flush=True)

    doc_ids, term_ids, scores = generate_data(num_docs, vocab_size, avg_nnz=150)
    half = num_docs // 2

    # Shard 1: docs [0, half)
    mask1 = doc_ids < half
    idx1 = build_index(doc_ids[mask1], term_ids[mask1], scores[mask1], vocab_size, device0)
    idx1_obj = dict_to_index(idx1)

    # Shard 2: docs [half, num_docs), remap to [0, half)
    mask2 = doc_ids >= half
    remapped_docs = doc_ids[mask2] - half
    idx2 = build_index(remapped_docs, term_ids[mask2], scores[mask2], vocab_size, device1)
    idx2_obj = dict_to_index(idx2)

    q_ids0, q_scores0 = make_queries(64, vocab_size, device=device0)
    q_ids1 = q_ids0.to(device1)
    q_scores1 = q_scores0.to(device1)

    print(f"  Shard 0: {idx1['num_docs']:,} docs on GPU 0", flush=True)
    print(f"  Shard 1: {idx2['num_docs']:,} docs on GPU 1", flush=True)

    # Single GPU baseline: just shard 1
    r_single = safe_run("Single GPU (1M docs)", lambda: triton_fused_score(idx1_obj, q_ids0, q_scores0, 10),
                         warmup=3, trials=10, sync_device=device0)

    # Multi-GPU: run both in parallel, merge
    def multi_gpu_search():
        # Launch on both GPUs concurrently via CUDA streams
        s0 = torch.cuda.Stream(device0)
        s1 = torch.cuda.Stream(device1)
        with torch.cuda.stream(s0):
            scores0, ids0 = triton_fused_score(idx1_obj, q_ids0, q_scores0, 10)
        with torch.cuda.stream(s1):
            scores1, ids1 = triton_fused_score(idx2_obj, q_ids1, q_scores1, 10)
        s0.synchronize()
        s1.synchronize()
        # Merge on CPU (fast for small top-k)
        scores1_cpu = scores1.cpu()
        ids1_cpu = ids1.cpu() + half  # remap doc IDs
        all_scores = torch.cat([scores0.cpu(), scores1_cpu], dim=1)
        all_ids = torch.cat([ids0.cpu(), ids1_cpu], dim=1)
        _, merge_idx = torch.topk(all_scores, k=10, dim=1)
        final_ids = torch.gather(all_ids, 1, merge_idx)
        final_scores = torch.gather(all_scores, 1, merge_idx)
        return final_scores, final_ids

    r_multi = safe_run("Multi-GPU (2M docs)", multi_gpu_search, warmup=3, trials=10, sync_device=None)

    results = {
        'single_gpu_1M': r_single,
        'multi_gpu_2M': r_multi,
        'speedup': round(r_single['mean_ms'] / r_multi['mean_ms'], 2) if 'mean_ms' in r_single and 'mean_ms' in r_multi else None,
    }

    del idx1, idx2, idx1_obj, idx2_obj
    gc.collect(); torch.cuda.empty_cache()
    return results


# =====================================================================
# EXPERIMENT 8: GPU utilization (compute vs memory bandwidth)
# =====================================================================
def exp_utilization(device):
    print("\n" + "="*70)
    print("EXPERIMENT 8: COMPUTE VS MEMORY BANDWIDTH ANALYSIS")
    print("="*70)
    results = {}
    vocab_size = 30522
    for num_docs in [100_000, 500_000, 1_000_000]:
        label = f"{num_docs//1000}K"
        print(f"\n--- {num_docs:,} documents ---", flush=True)
        gc.collect(); torch.cuda.empty_cache()

        doc_ids, term_ids, scores = generate_data(num_docs, vocab_size, avg_nnz=150)
        idx = build_index(doc_ids, term_ids, scores, vocab_size, device)
        idx_obj = dict_to_index(idx)
        q_ids, q_scores = make_queries(64, vocab_size, device=device)

        r = safe_run("Triton Fused", lambda: triton_fused_score(idx_obj, q_ids, q_scores, 10),
                      warmup=3, trials=10, sync_device=device)

        if 'mean_ms' in r:
            # Estimate bytes moved
            # Each query term loads a posting list: avg postings * 8 bytes (doc_id + score)
            avg_pl_len = num_docs * 150 / vocab_size  # avg postings per term
            n_query_terms = 64 * 30  # 64 queries, ~30 terms each
            bytes_read = n_query_terms * avg_pl_len * 8  # doc_id (4B) + score (4B)
            bytes_written = 64 * num_docs * 4  # output score matrix (atomic adds)
            total_bytes = bytes_read + bytes_written
            bandwidth_gbps = total_bytes / (r['mean_ms'] / 1000) / 1e9
            # H100 theoretical: 3.35 TB/s HBM bandwidth
            bw_utilization = bandwidth_gbps / 3350 * 100

            results[label] = {
                'latency_ms': r['mean_ms'],
                'bytes_read_gb': round(bytes_read / 1e9, 2),
                'bytes_written_gb': round(bytes_written / 1e9, 2),
                'effective_bandwidth_gbps': round(bandwidth_gbps, 1),
                'hbm_utilization_pct': round(bw_utilization, 1),
            }
        else:
            results[label] = r

        del idx, idx_obj, doc_ids, term_ids, scores
        gc.collect(); torch.cuda.empty_cache()

    return results


def main():
    device = torch.device("cuda:0")
    print(f"GPU 0: {torch.cuda.get_device_name(device)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.device_count() > 1:
        print(f"GPU 1: {torch.cuda.get_device_name(torch.device('cuda:1'))}")

    all_results = {}

    all_results['exp1_scale'] = exp_scale(device)
    all_results['exp2_sparsity'] = exp_sparsity(device)
    all_results['exp3_vocab'] = exp_vocab(device)
    all_results['exp4_batch'] = exp_batch(device)
    all_results['exp5_topk'] = exp_topk(device)
    all_results['exp6_memory'] = exp_memory(device)
    all_results['exp7_multi_gpu'] = exp_multi_gpu()
    all_results['exp8_utilization'] = exp_utilization(device)

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tracker", "comprehensive_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
