#!/bin/bash
# GPUSparse: Reproduce all paper results
#
# Requirements:
#   - 1x NVIDIA H100 80GB (or A100 80GB)
#   - CUDA 12.x with Triton 2.1+
#   - Python 3.10+ with: torch, triton, numpy, ir_datasets, transformers
#   - ~40GB GPU memory for 500K document evaluation
#   - Internet access for ir_datasets to download MS MARCO
#
# Environment variables:
#   DATA_ROOT: Path to cache directory (default: project_root/cache)
#
# Outputs results to tracker/*.json

set -e

source ~/.bashrc
conda activate retrieval_research

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "=========================================="
echo "GPUSparse: Reproducing All Paper Results"
echo "=========================================="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo ""

mkdir -p tracker cache

# ============================================================
# Phase 1: MS MARCO evaluation at 100K scale
# ============================================================
echo "=== Phase 1: MS MARCO Eval @ 100K docs ==="
echo "Running: msmarco_eval.py --num_docs 100000"
python src/msmarco_eval.py --num_docs 100000 --num_queries 500
echo ""

# ============================================================
# Phase 2: MS MARCO evaluation at 500K scale
# ============================================================
echo "=== Phase 2: MS MARCO Eval @ 500K docs ==="
echo "Running: msmarco_eval.py --num_docs 500000"
python src/msmarco_eval.py --num_docs 500000 --num_queries 1000
echo ""

# ============================================================
# Phase 3: Kernel comparison (V1/V3/V4 + dense baseline)
# ============================================================
echo "=== Phase 3: Kernel Comparison Benchmark ==="
echo "Running: benchmark_v4.py"
python src/benchmark_v4.py
echo ""

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "All experiments complete."
echo "Results:"
echo "=========================================="
for f in tracker/*.json; do
    echo "  $f ($(wc -c < "$f") bytes)"
done
echo ""
echo "Key result files for paper tables:"
echo "  tracker/msmarco_eval_100000.json -> Table 1 (quality + latency @ 100K)"
echo "  tracker/msmarco_eval_500000.json -> Table 2 (quality + latency @ 500K)"
echo "  tracker/benchmark_v4.json        -> Table 3 (kernel comparison)"
