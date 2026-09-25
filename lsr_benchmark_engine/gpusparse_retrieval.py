#!/usr/bin/env python3
"""GPUSparse engine for the lsr-benchmark leaderboard.

Most of the shipped engines are CPU-only (DuckDB, numpy-exhaustive, PISA, PyTerrier,
kANNolo, Seismic, BMP, IOQP, and the vector-database backends), but the benchmark is
not GPU-free: `pytorch-naive --use_gpu` holds a sparse CSR document index on the
device and scores by cuSPARSE SpMM against a densified query batch. This engine is
therefore not the first GPU entry, it is a different GPU design, and the interesting
comparison is against that baseline under one measurement protocol.

Two differences drive it. First, we traverse an inverted index with warp-aligned
posting lists and a fused scatter-add kernel rather than calling SpMM, and we never
densify queries: `pytorch-naive` materializes a [batch, |V|] dense query block
(30,522 floats per query). Second, and more consequentially at scale, SpMM here
produces a dense [batch, N_docs] score matrix before selection, which is the term that
caps batch size (16.5 GiB at batch 500 on 8.8M passages). Our chunked path bounds that
to O(batch * chunk), which is what lets a large collection run at a large batch.

It follows the convention of step-03-retrieval-approaches/numpy-exhaustive: a
@retrieve_command entry point, tirex_tracker `tracking()` around the index build and
the retrieval separately, and a gzipped TREC run.

Retrieval is exact in both engines, so any effectiveness difference should be fp32
summation noise and the comparison is about efficiency. Scoring computes the true inner
product between each query and every document with a non-zero term overlap, so the
ranking matches an exhaustive dense reference up to fp32 summation order. Notably we do NOT truncate queries to a
fixed number of terms: doing so is a silent approximation, and on MS MARCO dev the
maximum query has 107 non-zero terms against a mean of 44.8, so a top-64 cut changes
results for 13.4% of queries.
"""
import gzip
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings

# The kernels live in the GPUSparse package. In the container these are copied to
# /gpusparse; locally, point GPUSPARSE_SRC at the repo's src directory.
sys.path.insert(0, os.environ.get("GPUSPARSE_SRC", "/gpusparse"))
from triton_kernel import triton_fused_score                      # noqa: E402
from triton_kernel_chunked import build_chunk_boundaries           # noqa: E402
from fused_topk import triton_chunked_score_fused                  # noqa: E402
from gpu_inverted_index import GPUInvertedIndex                    # noqa: E402

BLOCK = 32          # posting lists are padded to a warp boundary
VOCAB_PAD = 1       # vocabulary is sized from the data, plus one


def to_csr(embeddings, dim):
    """Sparse embeddings -> (ids, scipy CSR). Avoids ever densifying."""
    ids, indptr, indices, data = [], [0], [], []
    for emb_id, tokens, values in embeddings:
        ids.append(emb_id)
        indices.extend(int(t) for t in tokens)
        data.extend(float(v) for v in values)
        indptr.append(len(indices))
    m = sp.csr_matrix(
        (np.asarray(data, dtype=np.float32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(ids), dim))
    return ids, m


def determine_dimension(*embedding_sets):
    d = 0
    for embeddings in embedding_sets:
        for _, tokens, _ in embeddings:
            if tokens:
                d = max(d, max(int(t) for t in tokens) + 1)
    return d + VOCAB_PAD


def build_index(doc_csr, device):
    """Term-major GPU inverted index with warp-aligned, doc_id-sorted posting lists."""
    num_docs, vocab = doc_csr.shape
    coo = doc_csr.tocoo()
    # kind='stable' matters: the kernel and any future merge-join rely on posting
    # lists being ascending in doc_id, and the default quicksort does not preserve it
    order = np.argsort(coo.col, kind="stable")
    terms, docs, scores = coo.col[order], coo.row[order], coo.data[order]

    lengths = np.bincount(terms, minlength=vocab).astype(np.int32)
    padded = ((lengths + BLOCK - 1) // BLOCK) * BLOCK
    offsets = np.zeros(vocab, dtype=np.int64)
    if vocab > 1:
        np.cumsum(padded[:-1], out=offsets[1:])

    total = int(offsets[-1] + padded[-1])
    all_docs = np.full(total, -1, dtype=np.int32)
    all_scores = np.zeros(total, dtype=np.float32)
    pos = 0
    for t in np.nonzero(lengths)[0]:
        L = lengths[t]
        o = offsets[t]
        all_docs[o:o + L] = docs[pos:pos + L]
        all_scores[o:o + L] = scores[pos:pos + L]
        pos += L

    idx = GPUInvertedIndex.__new__(GPUInvertedIndex)
    idx.doc_ids = torch.from_numpy(all_docs).to(device)
    idx.scores = torch.from_numpy(all_scores).to(device)
    idx.offsets = torch.from_numpy(offsets).to(device)
    idx.lengths = torch.from_numpy(lengths).to(device)
    idx.num_docs = int(num_docs)
    idx.vocab_size = int(vocab)
    idx.device = device
    return idx


def to_query_tensors(q_csr, device):
    """Dense [nq, max_nnz] term-id/score tensors. No truncation: max_nnz is the
    observed maximum, so every query term is scored."""
    nq = q_csr.shape[0]
    max_nnz = int(np.diff(q_csr.indptr).max()) if nq else 1
    ti = np.full((nq, max_nnz), -1, dtype=np.int32)
    ts = np.zeros((nq, max_nnz), dtype=np.float32)
    for i in range(nq):
        s, e = q_csr.indptr[i], q_csr.indptr[i + 1]
        n = e - s
        ti[i, :n] = q_csr.indices[s:e]
        ts[i, :n] = q_csr.data[s:e]
    return (torch.from_numpy(ti).to(device),
            torch.from_numpy(ts).to(device),
            max_nnz)


def retrieve(index, q_ids, q_ti, q_ts, doc_ids, k, batch_size=200, doc_chunk=131072):
    """Chunked scoring with threshold-compacted selection.

    The monolithic path allocates a dense [batch, num_docs] accumulator, which is what
    caps batch size on large collections: at batch 200 it is 6.6 GiB per million
    documents. Chunked scoring bounds it to [batch, doc_chunk] plus a running top-k, so
    memory no longer depends on collection size, and it is also faster here because
    selection runs over doc_chunk columns rather than all of them. The two paths retrieve
    the same top-1000 set: verified on the full 8.84M MS MARCO corpus and on the BEIR
    collections above one chunk. Ordering within the top-10 can differ for documents whose
    scores fall within about 1e-5 of each other, but so does the monolithic path against
    itself, because atomic accumulation makes the summation order irreproducible.

    Collections smaller than one chunk gain nothing from chunking and pay a small
    overhead, so those use the monolithic path directly.
    """
    results = []
    n = q_ti.shape[0]
    kk = min(k, index.num_docs)
    chunked = index.num_docs > doc_chunk
    bounds = build_chunk_boundaries(index, doc_chunk) if chunked else None

    for s0 in range(0, n, batch_size):
        s1 = min(s0 + batch_size, n)
        qi = q_ti[s0:s1].contiguous()
        qs = q_ts[s0:s1].contiguous()
        if chunked:
            scores, ids = triton_chunked_score_fused(
                index, qi, qs, top_k=kk, doc_chunk=doc_chunk, chunk_bounds=bounds)
        else:
            scores, ids = triton_fused_score(index, qi, qs, top_k=kk)
        scores = scores.cpu().numpy()
        ids = ids.cpu().numpy()
        for r in range(s1 - s0):
            ranking = [(q_ids[s0 + r], float(scores[r, j]), doc_ids[ids[r, j]])
                       for j in range(kk) if scores[r, j] > 0]
            results.append(ranking)
    return results


@retrieve_command()
def main(dataset, embedding, output, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("GPUSparse requires a CUDA device; none was visible.")

    register_metadata({
        "actor": {"team": "gpusparse"},
        "tag": f"gpusparse-{embedding.replace('/', '-')}-{k}",
        "description": (
            "Exact GPU-native learned sparse retrieval. Term-major inverted index "
            "resident in GPU memory with warp-aligned posting lists; a fused Triton "
            "kernel performs batched scatter-add scoring, chunked over the document space "
            "with threshold-compacted top-k so score memory is independent of "
            "collection size. No query "
            "pruning and no posting-list pruning, so rankings match an exhaustive "
            "dense reference up to fp32 summation order. "
            f"Device: {torch.cuda.get_device_name(0)}."),
    })

    print("Loading embeddings...", flush=True)
    d_emb = load_embeddings(dataset, embedding, "doc")
    q_emb = load_embeddings(dataset, embedding, "query")
    dim = determine_dimension(d_emb, q_emb)
    print(f"Vocabulary dimension: {dim}", flush=True)

    with tracking(export_file_path=output / "index-metadata.yml",
                  export_format=ExportFormat.IR_METADATA):
        doc_ids, doc_csr = to_csr(d_emb, dim)
        index = build_index(doc_csr, device)
        torch.cuda.synchronize()
    print(f"Indexed {index.num_docs} docs, {int(doc_csr.nnz)} postings", flush=True)

    q_ids, q_csr = to_csr(q_emb, dim)
    q_ti, q_ts, max_nnz = to_query_tensors(q_csr, device)
    print(f"{len(q_ids)} queries, max {max_nnz} terms (untruncated)", flush=True)

    with tracking(export_file_path=output / "retrieval-metadata.yml",
                  export_format=ExportFormat.IR_METADATA):
        results = retrieve(index, q_ids, q_ti, q_ts, doc_ids, k)
        torch.cuda.synchronize()

    with gzip.open(output / "run.txt.gz", "wt") as f:
        for ranking in results:
            for rank, (qid, score, docno) in enumerate(ranking, start=1):
                f.write(f"{qid} Q0 {docno} {rank} {score} gpusparse\n")


if __name__ == "__main__":
    main()
