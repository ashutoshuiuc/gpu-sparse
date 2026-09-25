# Corrected results, August 2026

These supersede several files in `../` and `../../final_results/`. Each superseded file
now carries a `_SUPERSEDED` key naming the defect and pointing here.

| file | what it contains |
|---|---|
| `seismic_heapfactor_sweep.json` | Seismic swept over its documented `heap_factor` range [0.7, 1.0]. Recall@1000 = 0.953, not 0.738. The earlier 0.738 came from `heap_factor=10.0`, outside the defined domain (0,1). |
| `component_breakdown.json` | Score-buffer alloc vs posting-list traversal vs top-k, timed in isolation with CUDA events across N and B. Traversal 55-80%, top-k 18-38%, alloc 2-7%. |
| `exactness_and_spare.json` | `max_terms` sweep proving bit-exact retrieval at 128, plus the three-way SPARe decomposition (as-published / host-side indptr / batched). |
| `profiling_and_cpu_baselines.json` | Nsight Compute counters for both kernels, the nsys kernel-time breakdown including all 8 `torch.topk` kernels, and CPU baselines (Seismic core-pinning curve, BMP). |

## Measurement conventions used here

- **Latency** is measured with CUDA events, never with Nsight Compute. `ncu` inflates
  wall time roughly 1.56x by serializing and replaying kernels, so its
  `gpu__time_duration` must not be read as latency.
- **Traffic, occupancy, coalescing and cache behaviour** come from `ncu` counters.
  Analytic byte models were the source of the two bandwidth errors corrected here.
- **CPU baselines** are reported with CPU time alongside wall time, and under explicit
  `taskset` pinning, because the Seismic library ignores `num_threads` and saturates
  34-36 cores by default. An unpinned wall-clock CPU-versus-GPU comparison is not an
  efficiency statement.
- `nproc` misreports 2 on these nodes despite 96 usable cores; verified by a
  process-scaling test (65x speedup at 64 processes).
