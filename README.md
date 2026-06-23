# GPUSparse

GPU kernels for exact learned sparse retrieval (SPLADE) on a GPU-resident inverted index.

Learned sparse retrieval is usually served on the CPU with inverted-index traversal
algorithms such as WAND and Block-Max WAND, whose pruning logic is hard to parallelize.
GPUSparse scores many queries at once by treating sparse retrieval as a batched
scatter-add over a GPU-resident inverted index. For each query term it gathers that
term's posting list and adds the weighted contributions into per-document score
accumulators, then selects the top-k. The scoring is exact: every posting entry is
processed and nothing is pruned.

The same scatter-add reformulation is used by SPARe's `iterative` mode. The contribution
here is the realization rather than the reformulation: a single fused Triton kernel
instead of a per-term Python loop, a warp-aligned and block-padded posting-list layout
for coalesced access, and an analysis of the work-efficiency versus bandwidth-efficiency
tradeoff. On the same SPLADE data, the fused kernel runs 23 to 270 times faster than a
faithful reimplementation of SPARe's iterative path, with identical top-10 rankings.

The system scales to the full 8.8M-passage MS MARCO collection on a single H100 (an 8.5 GB
index) and matches CPU exact scoring to three decimals (MRR@10 of 0.383, equal to Pyserini
SPLADE).

This repository contains the kernels, baselines, and benchmark scripts. It does not
contain the paper.

## Layout

```
src/
  triton_kernel.py                scatter-add scoring kernel (one program per query-term pair)
  gpu_inverted_index.py           GPU inverted index with block-padded posting lists
  run_fullscale_8m.py             full 8.8M MS MARCO build, retrieval, latency
  run_correctness_verification.py GPU vs CPU dense-matmul ranking agreement
  msmarco_eval.py                 MRR@10, nDCG@10, Recall with official qrels
baselines/
  run_pyserini_splade.py          Pyserini exact CPU SPLADE (ground truth)
  run_seismic_multithread.py      Seismic multi-threaded approximate baseline
  bench_spare_iterative.py        comparison against SPARe's iterative scatter-add
final_results/                    measured JSON outputs behind the reported numbers
```

## Requirements

An NVIDIA GPU with CUDA. Results were measured on an H100 80GB with CUDA 12.4,
Python 3.11, PyTorch 2.6, and Triton 3.2.

```
pip install -r requirements.txt
```

## Data

Pre-encoded SPLADE document vectors are not committed because of their size. Point the
scripts at a local copy through environment variables:

```
export SEISMIC_DATA=/path/to/seismic_data    # pre-encoded SPLADE documents
export RESULTS_DIR=/path/to/results          # output directory (defaults to ./results)
```

CSR caches for the 100K, 500K, and 1M scales are built on first run into `data_cache/`.

## Quick start

```
# Full-scale 8.8M MS MARCO: build the index, retrieve, measure latency
python src/run_fullscale_8m.py

# Correctness: GPU scatter-add vs CPU dense-matmul ground truth
python src/run_correctness_verification.py
```

## Notes

Scoring is exact by construction. Recall@1000 is at least 0.999 against dense matmul; the
small residual comes from floating-point tie-breaking when documents share near-identical
scores at the top-k boundary, because GPU atomic accumulation orders additions differently
from a sequential CPU sum.

The Seismic and Pyserini baselines need their own environments (`pyseismic-lsr` and
`pyserini`).
