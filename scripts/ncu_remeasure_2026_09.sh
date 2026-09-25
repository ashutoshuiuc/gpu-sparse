#!/bin/bash
# Re-collect Nsight Compute counters for both GPUSparse kernels at B=500.
#
# Why: the paper reports the doc-parallel kernel at "48.9 GB per 500-query batch" tagged
# measured, but the stored counter was taken at B=50 and multiplied by ten, and the table
# caption simultaneously calls that row an analytic model. This collects it at B=500.
#
# ncu inflates wall time by serializing and replaying kernels, so its timings are never
# read as latency. Byte counts, occupancy, coalescing and cache rates are the outputs.
set -u
export PATH=/opt/share/cuda-12.8/bin:$PATH
cd "$(dirname "$0")/.."
OUTDIR=${1:-./final_results/remeasure_2026_09}
mkdir -p "$OUTDIR"

METRICS=dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__waves_per_multiprocessor,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,\
lts__t_sector_hit_rate.pct,gpu__time_duration.sum

for WL in scatter docparallel; do
  echo "############ ncu counters: $WL, B=500 ############"
  ncu -k regex:"(scatter_add|doc_csr)" -c 3 --target-processes all \
      --csv --metrics "$METRICS" \
      python src/remeasure_2026_09.py --only counter --counter-workload "$WL" \
      > "$OUTDIR/ncu_${WL}.csv" 2> "$OUTDIR/ncu_${WL}.stderr"
  echo "--- rows: $(grep -c '^"' "$OUTDIR/ncu_${WL}.csv") ---"
  head -4 "$OUTDIR/ncu_${WL}.csv"
  tail -5 "$OUTDIR/ncu_${WL}.stderr"
done

echo "############ nsys: full pipeline kernel breakdown including every topk kernel ############"
nsys profile -o "$OUTDIR/nsys_scatter" --force-overwrite true --trace cuda \
    python src/remeasure_2026_09.py --only counter --counter-workload scatter \
    > "$OUTDIR/nsys_scatter.log" 2>&1
nsys stats --report cuda_gpu_kern_sum "$OUTDIR/nsys_scatter.nsys-rep" \
    | tee "$OUTDIR/nsys_scatter_kern_sum.txt"
