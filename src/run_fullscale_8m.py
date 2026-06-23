"""
GPUSparse: Full 8.8M MS MARCO evaluation.

Loads pre-encoded SPLADE vectors from Seismic's HuggingFace dataset,
builds GPU inverted index, runs retrieval on ALL 6,980 dev-small queries,
and computes MRR@10, nDCG@10, Recall@1000 with official qrels.
"""

import torch
import numpy as np
import time
import json
import os
import sys
import gc
import tarfile
from pathlib import Path
from collections import defaultdict
from math import log2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.triton_kernel import triton_fused_score
from src.gpu_inverted_index import GPUInvertedIndex

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

import os
# Pre-encoded SPLADE document data. Override with the SEISMIC_DATA env var;
# defaults to ./seismic_data under the repo root.
SEISMIC_DATA = Path(os.environ.get("SEISMIC_DATA", str(BASE / "seismic_data")))
CACHE = BASE / "data_cache"
CACHE.mkdir(exist_ok=True)

VOCAB_SIZE = 30522
DEVICE = "cuda:0"


def load_splade_from_tar(tar_path, vocab, max_docs=None):
    """Load SPLADE sparse vectors from Seismic's tar.gz format.
    Uses chunked numpy arrays to avoid Python list memory overhead.
    Vector keys are WordPiece token strings - we convert to vocab IDs.
    Returns: doc_ids_list, rows (np.int32), cols (np.int32), vals (np.float32)
    """

    print(f"Loading SPLADE vectors from {tar_path}...", flush=True)
    print(f"  Tokenizer vocab size: {len(vocab)}", flush=True)
    t0 = time.time()

    CHUNK_SIZE = 50_000_000
    row_chunks = []
    col_chunks = []
    val_chunks = []

    cur_rows = np.empty(CHUNK_SIZE, dtype=np.int32)
    cur_cols = np.empty(CHUNK_SIZE, dtype=np.int32)
    cur_vals = np.empty(CHUNK_SIZE, dtype=np.float32)
    chunk_pos = 0

    doc_ids_list = []
    doc_idx = 0
    skipped_tokens = 0

    with tarfile.open(tar_path, 'r:gz') as tar:
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
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = str(doc.get('id', doc.get('_id', doc.get('docid', doc_idx))))
                vector = doc.get('vector', {})
                if not isinstance(vector, dict) or len(vector) == 0:
                    continue

                for token, value in vector.items():
                    term_idx = vocab.get(token, -1)
                    if term_idx < 0 or term_idx >= VOCAB_SIZE:
                        skipped_tokens += 1
                        continue
                    if chunk_pos >= CHUNK_SIZE:
                        row_chunks.append(cur_rows[:chunk_pos].copy())
                        col_chunks.append(cur_cols[:chunk_pos].copy())
                        val_chunks.append(cur_vals[:chunk_pos].copy())
                        chunk_pos = 0
                    cur_rows[chunk_pos] = doc_idx
                    cur_cols[chunk_pos] = term_idx
                    cur_vals[chunk_pos] = float(value)
                    chunk_pos += 1

                doc_ids_list.append(doc_id)
                doc_idx += 1

                if doc_idx % 500000 == 0:
                    elapsed = time.time() - t0
                    print(f"  Loaded {doc_idx:,} docs ({elapsed:.0f}s, "
                          f"{doc_idx/elapsed:.0f} docs/s)", flush=True)

                if max_docs and doc_idx >= max_docs:
                    break
            if max_docs and doc_idx >= max_docs:
                break

    if chunk_pos > 0:
        row_chunks.append(cur_rows[:chunk_pos].copy())
        col_chunks.append(cur_cols[:chunk_pos].copy())
        val_chunks.append(cur_vals[:chunk_pos].copy())

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    vals = np.concatenate(val_chunks)
    del row_chunks, col_chunks, val_chunks, cur_rows, cur_cols, cur_vals

    elapsed = time.time() - t0
    print(f"  Done: {doc_idx:,} docs, {len(rows):,} postings in {elapsed:.0f}s", flush=True)
    print(f"  Avg terms/doc: {len(rows)/doc_idx:.1f}, skipped tokens: {skipped_tokens:,}", flush=True)

    return doc_ids_list, rows, cols, vals


def load_queries_from_tar(tar_path, vocab=None):
    """Load queries from Seismic's tar.gz format.
    Vector keys are WordPiece token strings - convert using vocab dict.
    """
    print(f"Loading queries from {tar_path}...")
    queries = {}
    with tarfile.open(tar_path, 'r:gz') as tar:
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
                qid = str(q.get('id', q.get('_id', q.get('qid', len(queries)))))
                vector = q.get('vector', {})
                if isinstance(vector, dict) and len(vector) > 0:
                    converted = {}
                    for token, value in vector.items():
                        tid = vocab.get(token, -1) if vocab else -1
                        if 0 <= tid < VOCAB_SIZE:
                            converted[tid] = float(value)
                    if converted:
                        queries[qid] = converted
    print(f"  Loaded {len(queries)} queries")
    return queries


def load_qrels(qrels_path):
    """Load MS MARCO qrels."""
    qrels = defaultdict(dict)
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
                qrels[qid][did] = rel
    print(f"  Loaded qrels for {len(qrels)} queries")
    return qrels


def build_gpu_inverted_index(rows, cols, vals, num_docs, vocab_size, device):
    """Build warp-aligned GPU inverted index from COO arrays."""
    BLOCK = 32
    t0 = time.time()

    sort_idx = np.argsort(cols)
    sorted_terms = cols[sort_idx]
    sorted_docs = rows[sort_idx]
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

    elapsed = time.time() - t0
    index_mb = total_padded * 8 / 1e6
    actual_mb = len(rows) * 8 / 1e6
    pad_overhead = (total_padded - len(rows)) / len(rows) * 100

    print(f"  Index built in {elapsed:.1f}s")
    print(f"  Total entries: {total_padded:,} (padded), {len(rows):,} (actual)")
    print(f"  Padding overhead: {pad_overhead:.1f}%")
    print(f"  Index size: {index_mb:.0f} MB")
    print(f"  Avg posting list length: {len(rows)/len(unique_terms):.1f}")

    idx = GPUInvertedIndex(
        doc_ids=torch.from_numpy(all_doc_ids).to(device),
        scores=torch.from_numpy(all_scores_flat).to(device),
        offsets=torch.from_numpy(offsets).to(device),
        lengths=torch.from_numpy(lengths).to(device),
        padded_lengths=torch.from_numpy(padded_lengths).to(device),
        max_scores=torch.from_numpy(max_scores).to(device),
        num_docs=num_docs,
        vocab_size=vocab_size,
        device=device,
    )

    return idx, {
        'index_mb': index_mb,
        'actual_mb': actual_mb,
        'padding_overhead_pct': pad_overhead,
        'build_time_s': elapsed,
        'total_entries': total_padded,
        'num_unique_terms': len(unique_terms),
        'avg_posting_len': len(rows) / len(unique_terms),
    }


def prepare_query_tensors(queries_dict, max_terms=64, device="cuda:0"):
    """Convert query dict {qid: {term_idx: weight}} to padded tensors."""
    qids = list(queries_dict.keys())
    n_queries = len(qids)

    term_ids = np.full((n_queries, max_terms), -1, dtype=np.int32)
    term_scores = np.zeros((n_queries, max_terms), dtype=np.float32)

    terms_per_query = []
    for i, qid in enumerate(qids):
        vec = queries_dict[qid]
        sorted_terms = sorted(vec.items(), key=lambda x: -x[1])[:max_terms]
        for j, (tid, score) in enumerate(sorted_terms):
            if tid < VOCAB_SIZE:
                term_ids[i, j] = tid
                term_scores[i, j] = score
        terms_per_query.append(len(vec))

    print(f"  Query stats: avg {np.mean(terms_per_query):.1f} terms, "
          f"max {np.max(terms_per_query)}, min {np.min(terms_per_query)}")

    return (qids,
            torch.from_numpy(term_ids).to(device),
            torch.from_numpy(term_scores).to(device))


def compute_metrics(rankings, qrels, doc_ids_list):
    """Compute MRR@10, nDCG@10, Recall@1000."""
    mrr_at_10 = []
    ndcg_at_10 = []
    recall_at_1000 = []

    for qid, ranked_doc_indices in rankings.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]

        # Convert internal indices to doc IDs
        ranked_doc_ids = [doc_ids_list[idx] for idx in ranked_doc_indices[:1000]]

        # MRR@10
        rr = 0.0
        for rank, did in enumerate(ranked_doc_ids[:10]):
            if did in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrr_at_10.append(rr)

        # nDCG@10
        dcg = 0.0
        for rank, did in enumerate(ranked_doc_ids[:10]):
            if did in relevant:
                dcg += relevant[did] / log2(rank + 2)
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum(r / log2(i + 2) for i, r in enumerate(ideal))
        ndcg_at_10.append(dcg / idcg if idcg > 0 else 0.0)

        # Recall@1000
        rel_set = set(relevant.keys())
        retrieved = set(ranked_doc_ids[:1000])
        recall_at_1000.append(len(rel_set & retrieved) / len(rel_set) if rel_set else 0.0)

    return {
        'mrr@10': float(np.mean(mrr_at_10)),
        'ndcg@10': float(np.mean(ndcg_at_10)),
        'recall@1000': float(np.mean(recall_at_1000)),
        'num_queries_evaluated': len(mrr_at_10),
    }


def run_retrieval(index, q_ids_tensor, q_scores_tensor, qids, top_k=1000,
                  batch_size=500, device="cuda:0"):
    """Run retrieval in batches and return rankings."""
    n_queries = len(qids)
    all_top_ids = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_q_ids = q_ids_tensor[start:end]
        batch_q_scores = q_scores_tensor[start:end]

        top_scores, top_doc_ids = triton_fused_score(
            index, batch_q_ids, batch_q_scores, top_k=top_k
        )
        all_top_ids.append(top_doc_ids.cpu())

        if (start // batch_size) % 5 == 0:
            print(f"  Retrieved {end}/{n_queries} queries")

    all_top_ids = torch.cat(all_top_ids, dim=0)

    rankings = {}
    for i, qid in enumerate(qids):
        rankings[qid] = all_top_ids[i].tolist()

    return rankings


def benchmark_latency(index, q_ids_tensor, q_scores_tensor, batch_sizes,
                      device="cuda:0", warmup=3, trials=5):
    """Benchmark latency at various batch sizes."""
    results = {}
    n_queries = q_ids_tensor.shape[0]

    for bs in batch_sizes:
        actual_bs = min(bs, n_queries)
        batch_q_ids = q_ids_tensor[:actual_bs]
        batch_q_scores = q_scores_tensor[:actual_bs]

        # Warmup
        for _ in range(warmup):
            triton_fused_score(index, batch_q_ids, batch_q_scores, top_k=10)
            torch.cuda.synchronize(device)

        # Timed runs
        times = []
        for _ in range(trials):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            triton_fused_score(index, batch_q_ids, batch_q_scores, top_k=10)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        per_query_us = mean_ms / actual_bs * 1000
        qps = actual_bs / (mean_ms / 1000)

        results[bs] = {
            'batch_size': actual_bs,
            'mean_ms': mean_ms,
            'std_ms': float(np.std(times)),
            'per_query_us': per_query_us,
            'qps': qps,
        }
        print(f"  Batch {bs}: {mean_ms:.2f}ms, {per_query_us:.0f}μs/query, {qps:.0f} QPS")

    return results


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    device = DEVICE
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    all_results = {}

    # Build tokenizer vocab (needed for both doc and query parsing)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.get_vocab()
    print(f"Tokenizer loaded: {len(vocab)} tokens")

    # Step 1: Load documents
    doc_tar = SEISMIC_DATA / "documents.tar.gz"
    cache_file = CACHE / "msmarco_8m_coo_v2.npz"
    doc_ids_cache = CACHE / "msmarco_8m_docids_v2.json"

    if cache_file.exists() and doc_ids_cache.exists():
        print("Loading cached 8.8M COO data...")
        data = np.load(cache_file)
        rows = data['rows']
        cols = data['cols']
        vals = data['vals']
        with open(doc_ids_cache) as f:
            doc_ids_list = json.load(f)
        print(f"  Loaded: {len(doc_ids_list):,} docs, {len(rows):,} postings")
    else:
        doc_ids_list, rows, cols, vals = load_splade_from_tar(doc_tar, vocab)
        print("Caching COO data...")
        np.savez(cache_file, rows=rows, cols=cols, vals=vals)
        with open(doc_ids_cache, 'w') as f:
            json.dump(doc_ids_list, f)
        print("  Cached.")

    num_docs = len(doc_ids_list)
    avg_terms_per_doc = len(rows) / num_docs
    all_results['corpus_stats'] = {
        'num_docs': num_docs,
        'total_postings': int(len(rows)),
        'avg_terms_per_doc': float(avg_terms_per_doc),
    }
    print(f"\nCorpus: {num_docs:,} docs, {len(rows):,} postings, "
          f"avg {avg_terms_per_doc:.1f} terms/doc")

    # Step 2: Load queries
    query_tar = SEISMIC_DATA / "queries.tar.gz"
    queries = load_queries_from_tar(query_tar, vocab=vocab)

    # Step 3: Load qrels
    qrels_path = SEISMIC_DATA / "qrels.dev.small.tsv"
    qrels = load_qrels(qrels_path)

    # Filter queries to those with qrels
    valid_queries = {qid: vec for qid, vec in queries.items() if qid in qrels}
    print(f"  Queries with qrels: {len(valid_queries)}")

    # Step 4: Build GPU inverted index
    print("\nBuilding GPU inverted index...")
    index, index_stats = build_gpu_inverted_index(
        rows, cols, vals, num_docs, VOCAB_SIZE, device
    )
    all_results['index_stats'] = index_stats

    gpu_mem_after_index = torch.cuda.memory_allocated(device) / 1e6
    print(f"  GPU memory after index: {gpu_mem_after_index:.0f} MB")
    all_results['gpu_memory_mb'] = gpu_mem_after_index

    # Step 5: Prepare query tensors
    print("\nPreparing query tensors...")
    qids, q_ids_tensor, q_scores_tensor = prepare_query_tensors(
        valid_queries, max_terms=64, device=device
    )

    # Step 6: Run retrieval (all queries, top-1000)
    print(f"\nRunning retrieval ({len(qids)} queries, top-1000)...")
    t0 = time.time()
    rankings = run_retrieval(index, q_ids_tensor, q_scores_tensor, qids,
                            top_k=1000, batch_size=500, device=device)
    retrieval_time = time.time() - t0
    print(f"  Total retrieval time: {retrieval_time:.1f}s")
    all_results['retrieval_time_s'] = retrieval_time

    # Step 7: Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(rankings, qrels, doc_ids_list)
    all_results['metrics'] = metrics
    print(f"  MRR@10: {metrics['mrr@10']:.4f}")
    print(f"  nDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"  Recall@1000: {metrics['recall@1000']:.4f}")
    print(f"  Queries evaluated: {metrics['num_queries_evaluated']}")

    # Step 8: Benchmark latency at various batch sizes
    print("\nBenchmarking latency...")
    latency_results = benchmark_latency(
        index, q_ids_tensor, q_scores_tensor,
        batch_sizes=[1, 8, 32, 128, 500, 1000],
        device=device
    )
    all_results['latency'] = latency_results

    # Step 9: Memory analysis
    total_gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6
    all_results['peak_gpu_memory_mb'] = total_gpu_mem
    print(f"\nPeak GPU memory: {total_gpu_mem:.0f} MB")

    # Save results
    output_path = RESULTS / "fullscale_8m_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
