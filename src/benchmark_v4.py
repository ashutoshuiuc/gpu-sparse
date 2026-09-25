"""
Benchmark V4 Doc-CSR kernel against V1/V3 scatter-add kernels.

Tests:
1. Correctness: V4 produces same scores as V1/V3
2. Bandwidth: Measure effective bandwidth utilization
3. Latency: Compare at 100K, 500K, 1M scale
4. Quality: MRR@10, nDCG@10 on MS MARCO dev set
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

RESULTS_DIR = Path(__file__).parent.parent / "tracker"
CACHE_DIR = Path(os.environ.get("DATA_ROOT", str(Path(__file__).parent.parent))) / "cache"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=100000)
    parser.add_argument("--num_queries", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Config: {args.num_docs} docs, {args.num_queries} queries, top-{args.top_k}")

    # Load cached SPLADE encodings
    cache_dir = CACHE_DIR / f"msmarco_{args.num_docs}"
    doc_cache = cache_dir / "doc_reps.pt"
    query_cache = cache_dir / "query_reps.pt"

    if not doc_cache.exists():
        print(f"ERROR: No cached encodings at {cache_dir}")
        print(f"Run msmarco_eval.py --num_docs {args.num_docs} first to generate encodings")
        sys.exit(1)

    print(f"Loading cached SPLADE encodings from {cache_dir}")
    doc_reps = torch.load(doc_cache, map_location="cpu", weights_only=True)
    query_reps = torch.load(query_cache, map_location="cpu", weights_only=True)

    if query_reps.shape[0] > args.num_queries:
        query_reps = query_reps[:args.num_queries]

    print(f"  Doc reps: {doc_reps.shape}")
    print(f"  Query reps: {query_reps.shape}")

    doc_nnz = (doc_reps > 0).sum(dim=1).float()
    query_nnz = (query_reps > 0).sum(dim=1).float()
    print(f"  Doc sparsity: avg {doc_nnz.mean():.1f} terms/doc")
    print(f"  Query sparsity: avg {query_nnz.mean():.1f} terms/query")

    # Load qrels for metrics
    import ir_datasets
    from collections import defaultdict
    ds = ir_datasets.load("msmarco-passage/dev/small")
    queries = {}
    for q in ds.queries_iter():
        queries[q.query_id] = q.text
    raw_qrels = defaultdict(dict)
    relevant_doc_ids = set()
    for qrel in ds.qrels_iter():
        raw_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        relevant_doc_ids.add(qrel.doc_id)

    # Need to recreate the doc_id mapping
    # Load doc IDs from cache or reconstruct
    docids_cache = cache_dir / "doc_ids.pt"
    if docids_cache.exists():
        doc_ids_list = torch.load(docids_cache, map_location="cpu", weights_only=True)
        if isinstance(doc_ids_list, torch.Tensor):
            doc_ids_list = doc_ids_list.tolist()
    else:
        # Reconstruct by loading data same way as msmarco_eval
        print("  Reconstructing doc ID mapping...")
        doc_ids_list = []
        filler_ids = []
        for doc in ds.docs_iter():
            if doc.doc_id in relevant_doc_ids:
                doc_ids_list.append(doc.doc_id)
            elif len(filler_ids) < (args.num_docs - len(relevant_doc_ids)):
                filler_ids.append(doc.doc_id)
            if (len(doc_ids_list) + len(filler_ids)) >= args.num_docs:
                if len(doc_ids_list) >= len(relevant_doc_ids):
                    break
        doc_ids_list = doc_ids_list + filler_ids[:args.num_docs - len(doc_ids_list)]

    docid_to_idx = {did: i for i, did in enumerate(doc_ids_list)}

    # Build qrels with internal indices
    qrels = defaultdict(dict)
    for qid in raw_qrels:
        for did, rel in raw_qrels[qid].items():
            if did in docid_to_idx:
                qrels[qid][docid_to_idx[did]] = rel

    valid_qids = [qid for qid in queries if qid in qrels and len(qrels[qid]) > 0]
    valid_qids = valid_qids[:args.num_queries]
    query_qrels = {qid: qrels[qid] for qid in valid_qids}

    print(f"  {len(valid_qids)} queries with relevant docs")

    # ==== Build indices ====
    print("\n=== Building Indices ===")

    # Term-centric index (for V1/V3)
    from src.msmarco_eval import build_gpu_index_from_dense, prepare_query_tensors
    print("Building term-centric index...")
    term_index = build_gpu_index_from_dense(doc_reps, device)

    # Doc-CSR index (for V4)
    from src.triton_kernel_v4 import build_doc_csr_index, build_query_weight_matrix
    print("Building doc-CSR index...")
    doc_csr_index = build_doc_csr_index(doc_reps, device)

    # Prepare queries
    q_term_ids, q_term_scores = prepare_query_tensors(query_reps, max_terms=128, device=device)
    query_weights = build_query_weight_matrix(query_reps, device)
    print(f"  Query weight matrix: {query_weights.shape}, {query_weights.nelement() * 4 / 1e6:.0f} MB")

    # ==== Benchmark V1 ====
    print("\n=== V1 Scatter-Add (BLOCK_PL=128) ===")
    from src.triton_kernel import triton_fused_score

    class IndexObj:
        pass
    idx_obj = IndexObj()
    for k in ['doc_ids', 'scores', 'offsets', 'lengths', 'max_scores', 'num_docs', 'vocab_size', 'device']:
        setattr(idx_obj, k, term_index[k])

    lat_v1, (scores_v1, ids_v1) = bench(
        lambda: triton_fused_score(idx_obj, q_term_ids, q_term_scores, top_k=args.top_k, block_pl=128),
        warmup=5, trials=20, device=device
    )
    print(f"  Latency: {lat_v1:.2f}ms ({lat_v1/len(valid_qids)*1000:.1f}us/query)")

    # ==== Benchmark V3 ====
    print("\n=== V3 Scatter-Add (BLOCK_PL=512) ===")
    from src.triton_kernel_v3 import triton_fused_score_v3

    lat_v3, (scores_v3, ids_v3) = bench(
        lambda: triton_fused_score_v3(term_index, q_term_ids, q_term_scores, top_k=args.top_k, block_pl=512),
        warmup=5, trials=20, device=device
    )
    print(f"  Latency: {lat_v3:.2f}ms ({lat_v3/len(valid_qids)*1000:.1f}us/query)")

    # ==== Benchmark V4 (Doc-CSR) with various BLOCK_D ====
    print("\n=== V4 Doc-CSR Kernel ===")
    from src.triton_kernel_v4 import triton_doc_csr_score

    best_v4_lat = float('inf')
    best_v4_config = None

    for block_t in [64, 128, 256, 512]:
        try:
            lat, (scores_v4, ids_v4) = bench(
                lambda bt=block_t: triton_doc_csr_score(
                    doc_csr_index, query_weights, top_k=args.top_k,
                    block_t=bt
                ),
                warmup=3, trials=10, device=device
            )
            if lat < best_v4_lat:
                best_v4_lat = lat
                best_v4_config = ('single', block_t)
            print(f"  BLOCK_T={block_t}: {lat:.2f}ms ({lat/len(valid_qids)*1000:.1f}us/query)")
        except Exception as e:
            print(f"  BLOCK_T={block_t}: FAILED ({str(e)[:200]})")

    if best_v4_config is None:
        print("\n  ERROR: All V4 configs failed!")
        sys.exit(1)

    print(f"\n  Best V4: {best_v4_config[0]}, BLOCK_T={best_v4_config[1]}, {best_v4_lat:.2f}ms")

    # Run best V4 for quality eval
    print(f"\n  Running best V4 config for quality metrics...")
    _, (scores_v4_best, ids_v4_best) = bench(
        lambda: triton_doc_csr_score(
            doc_csr_index, query_weights, top_k=args.top_k,
            block_t=best_v4_config[1]
        ),
        warmup=3, trials=5, device=device
    )

    # ==== Dense MatMul baseline ====
    print("\n=== Dense MatMul ===")
    doc_reps_gpu = doc_reps.to(device)
    query_reps_gpu = query_reps[:len(valid_qids)].to(device)

    try:
        lat_dense, (scores_dense, ids_dense) = bench(
            lambda: (lambda s: (s[0], s[1]))(torch.topk(torch.mm(query_reps_gpu, doc_reps_gpu.t()), k=min(args.top_k, args.num_docs), dim=1)),
            warmup=5, trials=20, device=device
        )
        print(f"  Latency: {lat_dense:.2f}ms ({lat_dense/len(valid_qids)*1000:.1f}us/query)")
    except Exception as e:
        print(f"  Dense MatMul failed: {e}")
        lat_dense = float('inf')

    del doc_reps_gpu, query_reps_gpu
    torch.cuda.empty_cache()

    # ==== Correctness check ====
    print("\n=== Correctness Check ===")
    # Compare V4 top-k against V1
    ids_v1_set = set()
    ids_v4_set = set()
    ids_v1_cpu = ids_v1.cpu().numpy()
    ids_v4_cpu = ids_v4_best.cpu().numpy()
    overlap_10 = 0
    overlap_100 = 0
    for q in range(min(100, len(valid_qids))):
        s1 = set(ids_v1_cpu[q, :10].tolist())
        s4 = set(ids_v4_cpu[q, :10].tolist())
        overlap_10 += len(s1 & s4) / 10
        s1_100 = set(ids_v1_cpu[q, :100].tolist())
        s4_100 = set(ids_v4_cpu[q, :100].tolist())
        overlap_100 += len(s1_100 & s4_100) / 100
    n = min(100, len(valid_qids))
    print(f"  V1 vs V4 overlap@10: {overlap_10/n:.4f}")
    print(f"  V1 vs V4 overlap@100: {overlap_100/n:.4f}")

    # ==== Quality Metrics ====
    print("\n=== Quality Metrics ===")
    from src.msmarco_eval import compute_metrics

    methods = {
        'V1 (scatter-add)': ids_v1,
        'V3 (scatter-add-512)': ids_v3,
        'V4 (doc-CSR)': ids_v4_best,
    }

    all_metrics = {}
    for name, ids_tensor in methods.items():
        ids_cpu = ids_tensor.cpu().numpy()
        scores_cpu = torch.zeros_like(ids_tensor).cpu().numpy()  # scores not needed for metrics
        run = {}
        for i, qid in enumerate(valid_qids):
            run[qid] = [(int(ids_cpu[i, j]), float(args.top_k - j)) for j in range(min(args.top_k, ids_cpu.shape[1]))]
        metrics = compute_metrics(run, query_qrels, k_values=[10, 100, 1000])
        all_metrics[name] = metrics
        print(f"\n  {name}:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # ==== Bandwidth Analysis ====
    print("\n=== Bandwidth Analysis ===")
    # V1/V3: reads posting list entries for each query term
    avg_qterms = float(query_nnz.mean())
    total_postings = int(term_index['lengths'].sum().item())
    avg_pl_len = total_postings / max(1, (term_index['lengths'] > 0).sum().item())
    bytes_per_batch_scatter = avg_qterms * avg_pl_len * 8 * len(valid_qids)

    # V4: reads all doc term entries for each query (once per doc)
    total_doc_entries = doc_csr_index['total_entries']
    bytes_per_batch_csr = total_doc_entries * 8 * 1  # read once, query lookup is in cache
    # Plus query weight lookups (avg doc_terms * 4 bytes per lookup)
    bytes_per_batch_csr += total_doc_entries * 4 * len(valid_qids) / len(valid_qids)  # per query

    print(f"  Scatter (V1/V3):")
    print(f"    Total bytes read: {bytes_per_batch_scatter/1e9:.2f} GB")
    bw_v1 = bytes_per_batch_scatter / (lat_v1 / 1000) / 1e9
    bw_v3 = bytes_per_batch_scatter / (lat_v3 / 1000) / 1e9
    print(f"    V1 bandwidth: {bw_v1:.1f} GB/s ({bw_v1/3350*100:.2f}% of peak)")
    print(f"    V3 bandwidth: {bw_v3:.1f} GB/s ({bw_v3/3350*100:.2f}% of peak)")

    print(f"  Doc-CSR (V4):")
    # For V4, each query reads ALL doc entries
    bytes_v4 = total_doc_entries * 8 * len(valid_qids)  # doc term data
    bytes_v4 += total_doc_entries * 4 * len(valid_qids)  # query weight lookups
    print(f"    Total bytes read: {bytes_v4/1e9:.2f} GB")
    bw_v4 = bytes_v4 / (best_v4_lat / 1000) / 1e9
    print(f"    V4 bandwidth: {bw_v4:.1f} GB/s ({bw_v4/3350*100:.2f}% of peak)")

    # ==== Summary ====
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Latency (ms)':>12} {'Per-Q (us)':>12} {'Speedup vs V1':>15}")
    print("-"*80)
    print(f"{'V1 scatter-add':<25} {lat_v1:>12.2f} {lat_v1/len(valid_qids)*1000:>12.1f} {'1.00x':>15}")
    print(f"{'V3 scatter-add-512':<25} {lat_v3:>12.2f} {lat_v3/len(valid_qids)*1000:>12.1f} {lat_v1/lat_v3:>14.2f}x")
    print(f"{'V4 doc-CSR (best)':<25} {best_v4_lat:>12.2f} {best_v4_lat/len(valid_qids)*1000:>12.1f} {lat_v1/best_v4_lat:>14.2f}x")
    if lat_dense < float('inf'):
        print(f"{'Dense MatMul':<25} {lat_dense:>12.2f} {lat_dense/len(valid_qids)*1000:>12.1f} {lat_v1/lat_dense:>14.2f}x")
    print("="*80)

    # Save results
    results = {
        'config': {
            'num_docs': args.num_docs,
            'num_queries': len(valid_qids),
            'top_k': args.top_k,
        },
        'v1': {
            'latency_ms': lat_v1,
            'per_query_us': lat_v1 / len(valid_qids) * 1000,
            'bandwidth_GBs': bw_v1,
        },
        'v3': {
            'latency_ms': lat_v3,
            'per_query_us': lat_v3 / len(valid_qids) * 1000,
            'bandwidth_GBs': bw_v3,
        },
        'v4_best': {
            'config': str(best_v4_config),
            'block_t': best_v4_config[1],
            'latency_ms': best_v4_lat,
            'per_query_us': best_v4_lat / len(valid_qids) * 1000,
            'bandwidth_GBs': bw_v4,
        },
        'dense_matmul': {
            'latency_ms': lat_dense if lat_dense < float('inf') else None,
        },
        'metrics': all_metrics,
        'correctness': {
            'v1_v4_overlap_at_10': overlap_10 / n,
            'v1_v4_overlap_at_100': overlap_100 / n,
        }
    }

    output_file = RESULTS_DIR / f"v4_benchmark_{args.num_docs}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
