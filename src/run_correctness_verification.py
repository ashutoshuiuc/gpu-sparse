"""
GPUSparse: Functional Correctness Verification.

Checks that GPU scoring matches CPU exact scoring up to floating-point tie-breaking.
At 100K/500K/1M: compare GPU Triton kernel vs CPU dense matmul.
Verify Recall@1000 >= 0.999 (residual is atomic-accumulation tie-breaking at the
top-k boundary, not a scoring error).
"""

import torch
import numpy as np
import time
import json
import sys
from pathlib import Path
from scipy import sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.triton_kernel import triton_fused_score
from src.gpu_inverted_index import GPUInvertedIndex

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
CACHE = BASE / "data_cache"
RESULTS.mkdir(exist_ok=True)

VOCAB_SIZE = 30522
DEVICE = "cuda:0"


def build_gpu_index_from_csr(csr_mat, device):
    """Build GPU inverted index from scipy CSR."""
    BLOCK = 32
    num_docs = csr_mat.shape[0]

    # Convert CSR to COO for inverted index (term-major)
    coo = csr_mat.tocoo()
    rows = coo.row.astype(np.int32)
    cols = coo.col.astype(np.int32)
    vals = coo.data.astype(np.float32)

    # Sort by term (column)
    sort_idx = np.argsort(cols)
    sorted_terms = cols[sort_idx]
    sorted_docs = rows[sort_idx]
    sorted_scores = vals[sort_idx]

    unique_terms, counts = np.unique(sorted_terms, return_counts=True)
    lengths = np.zeros(VOCAB_SIZE, dtype=np.int32)
    lengths[unique_terms] = counts
    padded_lengths = ((lengths + BLOCK - 1) // BLOCK) * BLOCK

    offsets = np.zeros(VOCAB_SIZE, dtype=np.int64)
    np.cumsum(padded_lengths[:-1], out=offsets[1:])

    total_padded = int(offsets[-1] + padded_lengths[-1])
    all_doc_ids = np.full(total_padded, -1, dtype=np.int32)
    all_scores_flat = np.zeros(total_padded, dtype=np.float32)
    max_scores = np.zeros(VOCAB_SIZE, dtype=np.float32)

    src_offset = 0
    for i, term_id in enumerate(unique_terms):
        n = counts[i]
        off = offsets[term_id]
        all_doc_ids[off:off+n] = sorted_docs[src_offset:src_offset+n]
        all_scores_flat[off:off+n] = sorted_scores[src_offset:src_offset+n]
        max_scores[term_id] = sorted_scores[src_offset:src_offset+n].max()
        src_offset += n

    return GPUInvertedIndex(
        doc_ids=torch.from_numpy(all_doc_ids).to(device),
        scores=torch.from_numpy(all_scores_flat).to(device),
        offsets=torch.from_numpy(offsets).to(device),
        lengths=torch.from_numpy(lengths).to(device),
        padded_lengths=torch.from_numpy(padded_lengths).to(device),
        max_scores=torch.from_numpy(max_scores).to(device),
        num_docs=num_docs,
        vocab_size=VOCAB_SIZE,
        device=device,
    )


def prepare_queries_from_meta(meta, max_terms=64, device="cuda:0"):
    """Prepare query tensors from cached metadata."""
    query_dense = meta["query_dense"]
    if isinstance(query_dense, torch.Tensor):
        query_dense = query_dense.numpy()

    n_queries = query_dense.shape[0]
    term_ids = np.full((n_queries, max_terms), -1, dtype=np.int32)
    term_scores = np.zeros((n_queries, max_terms), dtype=np.float32)

    for i in range(n_queries):
        nz_idx = np.nonzero(query_dense[i])[0]
        nz_vals = query_dense[i, nz_idx]
        sort_idx = np.argsort(-nz_vals)[:max_terms]
        n = len(sort_idx)
        term_ids[i, :n] = nz_idx[sort_idx]
        term_scores[i, :n] = nz_vals[sort_idx]

    return (torch.from_numpy(term_ids).to(device),
            torch.from_numpy(term_scores).to(device),
            query_dense)


def compute_recall_exact(gpu_ids, cpu_ids, k_values=[10, 100, 1000]):
    """Compute recall of GPU results vs CPU ground truth."""
    n_queries = gpu_ids.shape[0]
    results = {}
    for k in k_values:
        if k > gpu_ids.shape[1] or k > cpu_ids.shape[1]:
            continue
        matches = 0
        total = 0
        for qi in range(n_queries):
            gt_set = set(cpu_ids[qi, :k].tolist())
            pred_set = set(gpu_ids[qi, :k].tolist())
            matches += len(gt_set & pred_set)
            total += len(gt_set)
        results[f'recall@{k}'] = matches / total if total > 0 else 0.0
    return results


def main():
    device = DEVICE
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    all_results = {}

    scales = [100000, 500000, 1000000]

    for num_docs in scales:
        print(f"\n{'='*70}")
        print(f"SCALE: {num_docs:,} documents")
        print(f"{'='*70}")

        # Load cached CSR
        csr_path = CACHE / f"msmarco_splade_csr_{num_docs}.npz"
        meta_path = CACHE / f"msmarco_splade_csr_{num_docs}_meta.pt"

        if not csr_path.exists():
            print(f"  ERROR: Cache not found at {csr_path}")
            continue

        print("  Loading cached data...")
        doc_csr = sp.load_npz(csr_path)
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

        print(f"  Docs: {doc_csr.shape[0]}, NNZ: {doc_csr.nnz:,}")

        # Prepare queries
        q_ids_tensor, q_scores_tensor, query_dense = prepare_queries_from_meta(
            meta, max_terms=64, device=device
        )
        n_queries = q_ids_tensor.shape[0]
        print(f"  Queries: {n_queries}")

        # Method 1: CPU exact dense matmul (ground truth)
        print("  Computing CPU ground truth (dense matmul)...")
        doc_dense = torch.from_numpy(doc_csr.toarray().astype(np.float32))
        query_dense_torch = torch.from_numpy(query_dense.astype(np.float32))

        t0 = time.time()
        cpu_scores = query_dense_torch @ doc_dense.t()
        cpu_top_scores, cpu_top_ids = torch.topk(cpu_scores, k=1000, dim=1)
        cpu_time = time.time() - t0
        print(f"    CPU matmul: {cpu_time:.1f}s")

        del doc_dense, cpu_scores
        torch.cuda.empty_cache()

        # Method 2: GPU Triton kernel
        print("  Building GPU index...")
        gpu_index = build_gpu_index_from_csr(doc_csr, device)

        print("  Running GPU Triton kernel...")
        torch.cuda.synchronize(device)
        t0 = time.time()
        gpu_top_scores, gpu_top_ids = triton_fused_score(
            gpu_index, q_ids_tensor, q_scores_tensor, top_k=1000
        )
        torch.cuda.synchronize(device)
        gpu_time = time.time() - t0
        print(f"    GPU Triton: {gpu_time:.3f}s")

        # Compare results
        gpu_ids_cpu = gpu_top_ids.cpu()
        recall = compute_recall_exact(gpu_ids_cpu, cpu_top_ids,
                                      k_values=[10, 100, 1000])

        # Also check score differences
        gpu_scores_cpu = gpu_top_scores.cpu()
        score_diff = torch.abs(gpu_scores_cpu[:, 0] - cpu_top_scores[:, 0])
        max_score_diff = score_diff.max().item()
        mean_score_diff = score_diff.mean().item()

        scale_results = {
            'num_docs': num_docs,
            'num_queries': n_queries,
            'cpu_time_s': cpu_time,
            'gpu_time_s': gpu_time,
            'speedup': cpu_time / gpu_time,
            'recall': recall,
            'max_top1_score_diff': max_score_diff,
            'mean_top1_score_diff': mean_score_diff,
        }

        print(f"\n  RESULTS:")
        print(f"    Recall@10: {recall.get('recall@10', 'N/A'):.6f}")
        print(f"    Recall@100: {recall.get('recall@100', 'N/A'):.6f}")
        print(f"    Recall@1000: {recall.get('recall@1000', 'N/A'):.6f}")
        print(f"    Max top-1 score diff: {max_score_diff:.8f}")
        print(f"    Mean top-1 score diff: {mean_score_diff:.8f}")
        print(f"    Speedup: {cpu_time/gpu_time:.1f}x")

        if recall.get('recall@1000', 0) < 0.999:
            print("    WARNING: Recall@1000 < 0.999 - possible scoring bug!")

        all_results[str(num_docs)] = scale_results

        # Cleanup
        del gpu_index, gpu_top_scores, gpu_top_ids, cpu_top_ids, cpu_top_scores
        torch.cuda.empty_cache()

    # Save
    output_path = RESULTS / "correctness_verification.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
