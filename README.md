# GPUSparse

GPU kernels for exact learned sparse retrieval (SPLADE) on a GPU-resident
inverted index, and the measurements behind the paper "GPUSparse: Exact Learned
Sparse Retrieval on a GPU, and What It Costs".

Learned sparse retrieval is usually served on the CPU with inverted-index
traversal algorithms such as WAND and Block-Max WAND, whose pruning logic is hard
to parallelize. GPUSparse scores many queries at once by treating sparse retrieval
as a batched scatter-add over a GPU-resident inverted index: for each query term
it gathers that term's posting list and adds the weighted contributions into
per-document accumulators, then selects the top-k. The scoring is exact. Every
posting entry is processed and nothing is pruned.

The same scatter-add reformulation is used by SPARe's `iterative` mode. The
contribution here is the realization rather than the reformulation, plus the cost
accounting in the paper's title.

This repository is the artifact for the paper: source, configs, scripts and the
result files each number comes from. It does not contain the paper source.

## Read this before quoting any number

The paper was substantially revised in September 2026. Several first-version
figures were withdrawn, and two contribution claims were dropped outright. The
arXiv v2 "Comments" field lists all of them. The ones that matter to anyone
reading this code:

**1. Scoring is exact at recall 1.000, and it is not bitwise reproducible.** Both
halves matter. Against an exhaustive FP32 reference, Recall@10, @100 and @1000 are
all exactly 1.000 at 100K, 500K and 1M documents. The 0.999 figures in the first
version came from a harness that built query tensors at a fixed width of 64 terms,
truncating 67 of 500 queries (the longest MS MARCO SPLADE query has 107 terms).
Separately, two runs of the same input can differ by up to 3.81e-06 because
scatter-add order is nondeterministic, while top-10 and top-1000 agreement are
both 1.000. The word "bit-exact" was removed from the paper.

**2. The component-breakdown primacy claim is withdrawn.** Ding et al., WWW 2009,
Table 3 already gives a four-way split of GPU inverted-index latency, states its
CPU baselines are single-core, and concludes that a quad-core CPU was the better
value. What survives is narrower and more useful: our selection share *shrinks*
with scale, from 37.6% to 17.8%, which reconciles the disagreeing splits reported
by FAERY, AgentIR and FAISS.

**3. Our Seismic numbers in version 1 were wrong, and the error was ours.** Those
runs used `heap_factor=10.0`, outside the parameter's documented domain of (0,1),
which effectively disables block skipping. Correctly configured, Seismic reaches
Recall@1000 0.953 against the exhaustive reference's 0.983. The published
literature already had this right. See the measurement trap below, which is the
part worth carrying to other projects.

Also withdrawn: a 235x speedup over Pyserini (CPU query encoding was timed inside
the baseline and excluded from ours), a 270x speedup over SPARe (host-device syncs
in our own reimplementation of it), two bandwidth figures wrong by 42x and 1.56x
in opposite directions, which inverted a work-efficiency conclusion, a
collection statistic of 127.2 terms per document that matched no encoded
collection, a 1M latency produced by the wrong kernel, and a description of cuVS
as dense-only.

## The headline result is a cost result

One H100 is worth about seven CPU cores here. We are 5.6x faster than one pinned
core, reach parity between 4 and 8 cores, and are 2.8x slower than a 32-core
socket. An H100 costs far more than seven cores, so at equal quality the CPU is
the better value. That is Ding et al.'s 2009 conclusion and it still holds. Where
the GPU does earn its cost is the case they did not test: a hard exactness
requirement at large k.

## A measurement trap worth knowing about

`pyseismic-lsr` silently ignores its `num_threads` argument. In `src/pylib/mod.rs`
it builds a Rayon pool with `.num_threads(n).build()` and drops the result
immediately, with no `.install()` and no `.build_global()` anywhere in the file, so
the subsequent `par_bridge` runs on Rayon's global pool sized to all logical CPUs.
Passing `num_threads=1` and reporting the result as single-threaded made Seismic
look 3x faster than an H100 when the honest per-core figure is 5.6x slower. It
inverted the sign of a headline.

Two mechanisms are needed, and the scripts here set both: `RAYON_NUM_THREADS=1`,
which does control the global pool, and `taskset`, since `taskset` alone only
confines an oversubscribed pool rather than shrinking it. The diagnostic that
exposes it is to **report CPU seconds over wall seconds next to every CPU latency**:
a ratio near 1.0 means one core really did the work, and anything well above that
means the pin leaked and the row is void. `baselines/seismic_k_calibration.py`
does this per row.

A related caution: single-core CPU latency varied about 25% between the machines
we measured on, with quality metrics identical to four decimals. Do not compare
CPU latencies across machines, and do not quote one to three significant figures.

## Layout

```
src/
  triton_kernel.py            # the fused scatter-add scoring kernel (reported)
  triton_kernel_v2/v3/v4.py   # earlier variants kept for the ablation
  gpu_inverted_index.py       # GPU-resident index construction
  fused_topk.py               # threshold-compacted selection
  remeasure_2026_09.py        # bandwidth, counters, dtype-matched baselines
  settle_mrr_and_blockpl.py   # query-width and posting-list-block sweeps
  run_correctness_verification.py  # exactness against an exhaustive reference
  run_fullscale_8m.py         # the 8.8M-document run
  msmarco_eval.py             # official metric implementation
baselines/                    # Seismic, Pyserini, SPARe and BMP harnesses
lsr_benchmark_engine/         # engine plug-in used for the third-party comparison
scripts/                      # reproduction and profiling drivers
final_results/                # results; see PROVENANCE.md for which file backs which number
```

## Requirements

- An NVIDIA GPU. Every GPU number in the current paper was measured on a single
  H100 80GB HBM3 with CUDA 12.8.
- Python 3.13, PyTorch 2.9.1+cu128, Triton 3.5.1.
- CPU baselines need `pyseismic-lsr`, `pyserini` or `bmp` as applicable.

```bash
pip install -r requirements.txt
```

Paths are resolved from each script's own location, so any checkout works from any
working directory. Data and output locations are overridable:

| variable | meaning | default |
|---|---|---|
| `GPUSPARSE_DATA` | datasets, encoded collections, prebuilt baseline indexes | `<repo>/data` |
| `GPUSPARSE_DATA_SEISMIC` | prebuilt Seismic index | `<repo>/data/seismic` |
| `GPUSPARSE_RESULTS` | where re-run scripts write | `<repo>/results` |
| `RAYON_NUM_THREADS` | required for any single-core Seismic measurement | unset |
| `CONDA_ENV` | optional; if set, baseline scripts activate it | unset |

## Reproducing the paper

```bash
# Exactness, self-consistency, the posting-list ordering invariant, and quality
python src/run_correctness_verification.py

# Bandwidth, Nsight Compute counters, dtype-matched baselines
python src/remeasure_2026_09.py --only all

# Query-width and BLOCK_PL sweeps (settles MRR@10 and the block-size claim)
python src/settle_mrr_and_blockpl.py

# Single-core Seismic calibrated against its published figures, k in {10,100,1000}
RAYON_NUM_THREADS=1 taskset -c 0 python baselines/seismic_k_calibration.py
```

Measurement conventions the scripts enforce: latency from `torch.cuda.Event`,
medians rather than means; correctness checked against an exhaustive FP32
reference before any timing is believed; CPU baselines pinned with both mechanisms
above and reported with CPU time per query alongside wall latency. Counters come
only from Nsight Compute, whose wall times are inflated roughly 1.6x by kernel
replay and must never be read as latency. Any percent-of-peak-bandwidth figure
derived from an analytic byte count rather than from counters should be
distrusted; two such figures here were wrong by 42x and 1.56x.

## Settled numbers

- MRR@10 at 100K is 0.8916, measured at five query widths and identical to four
  decimals at all of them.
- `BLOCK_PL` 512 is fastest; 128 costs 1.2%; the spread from 64 to 512 is 6.8%,
  not the 15% once claimed. Every block size returns an identical top-10.
- Single-core Seismic at k=10, tuned inside its documented domain, is 500 us per
  query, inside its published 187 to 531 us range. The span from k=1000 to k=10 is
  17.6x, which accounts for the whole apparent discrepancy. k=10 costs recall
  (R@10 of 0.65), so it is not an operating point for exact retrieval.
- Measured and found not worth doing: a naive two-GPU split is a slowdown (5.67 to
  9.47 ms) because host coordination dominates sub-10 ms latencies, and GPU-side
  WAND upper-bound pruning gives no speedup on SPLADE because the threshold starts
  at zero.

## Citation

Please cite the paper rather than this repository. The arXiv v2 "Comments" field
records which claims changed and why.
