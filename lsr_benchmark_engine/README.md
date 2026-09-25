# GPUSparse engine for lsr-benchmark

An inverted-index GPU engine for the lsr-benchmark leaderboard. Exact learned sparse
retrieval: a term-major index resident in GPU memory with warp-aligned posting lists,
and a fused Triton kernel doing batched scatter-add scoring, chunked over the document
space with threshold-compacted top-k so that score memory does not grow with collection
size.

## Why this is worth adding

The benchmark's value is that it measures many engines under one protocol, and our own
experience is that CPU-versus-GPU comparisons are very easy to get wrong in both
directions: an unpinned wall-clock comparison made a CPU baseline look 3x faster than an
H100, while the same baseline measured per core was 5.6x slower. Putting this engine
under the harness removes that ambiguity.

Most shipped engines are CPU-only, but the benchmark is not GPU-free: `pytorch-naive
--use_gpu` keeps a sparse CSR index on device and scores by cuSPARSE SpMM against a
densified query batch. This is therefore not the first GPU entry but a different GPU
design, and that baseline is the most informative comparison. Two things differ. We
traverse an inverted index with a fused scatter-add kernel instead of calling SpMM, and
we never densify queries (`pytorch-naive` materializes a [batch, |V|] block, 30,522
floats per query). More consequentially at scale, SpMM produces a dense
[batch, num_docs] score matrix before selection, 16.5 GiB at batch 500 on 8.8M passages;
chunking bounds that to [batch, chunk], which is what allows a large collection at a
large batch. Both are exact, so effectiveness should agree to fp32 noise and the
comparison is about efficiency.

## Exactness

No query pruning and no posting-list pruning. Queries are NOT truncated to a fixed
term budget, which matters: on MS MARCO dev the mean query has 44.8 non-zero terms
but the maximum has 107, so a top-64 cut silently changes results for 13.4% of
queries. Rankings match an exhaustive dense reference up to floating-point tie-breaking.

Be precise about what that means: this is not bitwise determinism. Scatter-add accumulates
with atomic_add, so the summation order changes between runs. Two runs of the same kernel on
the same input give scores differing by up to 7.6e-06 and are not bitwise equal, which is
enough to reorder the roughly one document per query sitting within 1e-5 of the k-th best
score. The guarantee is that no document is skipped and every score is a full inner product,
not that a ranking is reproducible to the last position. The chunked and monolithic paths
retrieve the same top-1000 set, and their top-10 differences match the rate at which the
monolithic path disagrees with itself.

## Running it

    # locally, against the repo
    GPUSPARSE_SRC=../src python3 gpusparse_retrieval.py \
        --dataset <lsr-dataset> --embedding <model> --output out --k 1000

    # build for TIRA (amd64 only)
    cp -r ../src gpusparse_src
    docker build --platform linux/amd64 -t gpusparse-lsr .

Then submit with `tira-cli code-submission` and select a GPU resource tier
(`a100-resources-gpu` or `h100-resources-gpu`) when starting the run. The tier is
chosen at run time, not at submission time.

## Hardware reporting

Both phases are wrapped in `tirex_tracker.tracking()` and emit ir_metadata, so index
build and retrieval are attributed separately. Note that the published leaderboard
currently reports 0.0 J for retrieval energy on the CPU engines, which the authors'
own notes attribute to the tracker not running in privileged mode. If that is fixed,
GPU energy needs `nvidia-smi`-based accounting rather than RAPL, since RAPL covers
only the CPU package.
