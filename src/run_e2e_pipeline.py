"""
GPUSparse: End-to-End Pipeline Benchmark.

Measures SPLADE encoding + GPU scoring + top-k at 100K and 1M docs.
Multiple batch sizes, many runs, separately reports encoding vs scoring time.
"""

import torch
import numpy as np
import time
import json
import sys
import gc
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


def load_splade_model(device):
    """Load SPLADE encoder."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    model_name = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def encode_queries_splade(texts, tokenizer, model, device, max_length=128):
    """Encode query texts with SPLADE, return sparse tensors."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        mask = inputs["attention_mask"].unsqueeze(-1)
        sparse = torch.log1p(torch.relu(logits))
        sparse = (sparse * mask).max(dim=1).values  # [batch, vocab]
    return sparse


def sparse_to_query_tensors(sparse_reps, max_terms=64):
    """Convert dense SPLADE output to sparse (term_ids, term_scores) for Triton kernel."""
    batch_size = sparse_reps.shape[0]
    device = sparse_reps.device

    term_ids = torch.full((batch_size, max_terms), -1, dtype=torch.int32, device=device)
    term_scores = torch.zeros((batch_size, max_terms), dtype=torch.float32, device=device)

    for i in range(batch_size):
        nz_mask = sparse_reps[i] > 0
        nz_idx = nz_mask.nonzero(as_tuple=True)[0]
        nz_vals = sparse_reps[i, nz_idx]

        n_terms = min(len(nz_idx), max_terms)
        if n_terms > 0:
            top_vals, top_indices = torch.topk(nz_vals, n_terms)
            term_ids[i, :n_terms] = nz_idx[top_indices].int()
            term_scores[i, :n_terms] = top_vals

    return term_ids, term_scores


def build_gpu_index_from_csr(csr_mat, device):
    """Build GPU inverted index from scipy CSR."""
    BLOCK = 32
    num_docs = csr_mat.shape[0]
    coo = csr_mat.tocoo()
    rows = coo.row.astype(np.int32)
    cols = coo.col.astype(np.int32)
    vals = coo.data.astype(np.float32)

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


def main():
    device = DEVICE
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Load SPLADE model
    print("Loading SPLADE model...")
    tokenizer, model = load_splade_model(device)

    # Load sample queries (real MS MARCO queries from ir_datasets)
    import ir_datasets
    ds = ir_datasets.load("msmarco-passage/dev/small")
    query_texts = []
    for q in ds.queries_iter():
        query_texts.append(q.text)
        if len(query_texts) >= 200:
            break
    print(f"Loaded {len(query_texts)} query texts")

    all_results = {}

    for num_docs in [100000, 1000000]:
        print(f"\n{'='*70}")
        print(f"SCALE: {num_docs:,} documents")
        print(f"{'='*70}")

        # Load index
        csr_path = CACHE / f"msmarco_splade_csr_{num_docs}.npz"
        if not csr_path.exists():
            print(f"  Cache not found: {csr_path}")
            continue

        doc_csr = sp.load_npz(csr_path)
        gpu_index = build_gpu_index_from_csr(doc_csr, device)
        del doc_csr
        gc.collect()

        scale_results = {}
        batch_sizes = [1, 4, 8, 16, 32, 64, 128]

        for bs in batch_sizes:
            batch_texts = query_texts[:bs]

            # Warmup
            for _ in range(2):
                sparse_reps = encode_queries_splade(batch_texts, tokenizer, model, device)
                q_ids, q_scores = sparse_to_query_tensors(sparse_reps)
                triton_fused_score(gpu_index, q_ids, q_scores, top_k=10)
                torch.cuda.synchronize(device)

            # Timed runs
            encode_times = []
            score_times = []
            total_times = []

            for trial in range(10):
                torch.cuda.synchronize(device)

                # Encoding
                t0 = time.perf_counter()
                sparse_reps = encode_queries_splade(batch_texts, tokenizer, model, device)
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()

                # Sparse conversion + scoring
                q_ids, q_scores = sparse_to_query_tensors(sparse_reps)
                triton_fused_score(gpu_index, q_ids, q_scores, top_k=10)
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()

                encode_times.append((t1 - t0) * 1000)
                score_times.append((t2 - t1) * 1000)
                total_times.append((t2 - t0) * 1000)

            result = {
                'batch_size': bs,
                'encode_ms': {
                    'mean': float(np.mean(encode_times)),
                    'std': float(np.std(encode_times)),
                },
                'score_ms': {
                    'mean': float(np.mean(score_times)),
                    'std': float(np.std(score_times)),
                },
                'total_ms': {
                    'mean': float(np.mean(total_times)),
                    'std': float(np.std(total_times)),
                },
                'per_query_ms': float(np.mean(total_times) / bs),
                'e2e_qps': float(bs / (np.mean(total_times) / 1000)),
            }
            scale_results[bs] = result

            print(f"  Batch {bs:3d}: encode={result['encode_ms']['mean']:.2f}ms, "
                  f"score={result['score_ms']['mean']:.2f}ms, "
                  f"total={result['total_ms']['mean']:.2f}ms, "
                  f"per-query={result['per_query_ms']:.2f}ms, "
                  f"QPS={result['e2e_qps']:.0f}")

        all_results[str(num_docs)] = scale_results

        del gpu_index
        torch.cuda.empty_cache()
        gc.collect()

    # Save
    output_path = RESULTS / "e2e_pipeline_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
