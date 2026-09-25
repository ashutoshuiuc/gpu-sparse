#!/bin/bash
# GPUSparse: regenerate every number the paper reports.
#
# Requirements
#   - one NVIDIA H100 80GB for the GPU numbers
#   - Python 3.13, PyTorch 2.9.1+cu128, Triton 3.5.1 (see requirements.txt)
#   - CPU baselines additionally need pyseismic-lsr, pyserini or bmp
#   - an encoded SPLADE collection; see the data section of README.md
#
# Environment variables
#   GPUSPARSE_DATA          datasets and encoded collections (default <repo>/data)
#   GPUSPARSE_DATA_SEISMIC  prebuilt Seismic index (default <repo>/data/seismic)
#   GPUSPARSE_RESULTS       where results are written (default <repo>/results)
#   CONDA_ENV               optional; if set, this script activates it
#
# CPU BASELINE PINNING. Any single-core CPU measurement here needs both
# RAYON_NUM_THREADS=1 and taskset. pyseismic-lsr silently ignores its num_threads
# argument (it builds a Rayon pool and drops it without installing it), so
# num_threads alone measures the whole socket, and taskset alone only confines an
# oversubscribed pool rather than shrinking it. Every CPU row reports CPU seconds
# over wall seconds so a leaked pin is visible rather than silent: a ratio near 1.0
# means one core really did the work.

set -uo pipefail

# Resolve the repository root from this script's own location, so the script works
# from any checkout and any working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

OUT="${GPUSPARSE_RESULTS:-$REPO_ROOT/results}"
mkdir -p "$OUT"

FAILED=0
run() {                     # run <label> <command...>
    local label="$1"; shift
    echo "---------- $label ----------"
    if "$@"; then
        echo "[ok] $label"
    else
        echo "[FAILED] $label (exit $?)" >&2
        FAILED=$((FAILED + 1))
    fi
}

echo "=============================================="
echo "GPUSparse: reproducing all paper results"
echo "=============================================="
python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.version.cuda)"
python -c "import torch; print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo

echo "########## 1. exactness, self-consistency, and the ordering invariant ##########"
# Recall against an exhaustive FP32 reference, run-to-run agreement, and the
# posting-list ascending-order invariant the chunk sub-ranges depend on.
run "correctness" python src/run_correctness_verification.py

echo
echo "########## 2. bandwidth, counters, dtype-matched baselines ##########"
run "remeasure" python src/remeasure_2026_09.py --only all

echo
echo "########## 3. query-width and posting-list-block sweeps ##########"
# Settles MRR@10 (identical at five query widths) and the BLOCK_PL claim.
run "mrr_and_blockpl" python src/settle_mrr_and_blockpl.py

echo
echo "########## 4. full-scale run at 8.8M documents ##########"
run "fullscale_8m" python src/run_fullscale_8m.py

echo
echo "########## 5. CPU baselines, pinned to one core ##########"
# Both pinning mechanisms, for the reason in the header.
run "seismic_k_calibration" env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    taskset -c 0 python baselines/seismic_k_calibration.py
run "seismic_threads" python baselines/run_seismic_multithread.py

echo
echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
    echo "all sections completed; results under $OUT"
else
    echo "$FAILED section(s) FAILED; see the [FAILED] lines above" >&2
fi
echo "Nsight Compute counters are collected separately:"
echo "  bash scripts/ncu_remeasure_2026_09.sh \"\$OUT\""
echo "ncu wall times are inflated roughly 1.6x by kernel replay."
echo "Never read them as latency; use the CUDA-event numbers above."
echo "=============================================="
exit "$FAILED"
