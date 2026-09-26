# Which artifact backs which number

Written 2026-09-24 during the audit.

## Authoritative

`remeasure_2026_09/` (torch 2.9.1+cu128, triton 3.5.1, CUDA 12.8, H100 80GB HBM3):

- `results.json`: the exactness sweep that settles the paper's Recall claim
  (1.000 at query width >= 96, at 100K / 500K / 1M), the self-consistency bound
  (scores not bitwise equal, max deviation 3.81e-06, top-10 and top-1000
  agreement 1.000), the posting-list ascending invariant (0 non-ascending lists
  of 26,972 / 28,023 / 28,245 terms), and MRR@10 = 0.8916 at 100K.
- `ncu_scatter.csv`, `ncu_docparallel.csv`: Nsight Compute counters at B=500,
  the batch size the paper reports. Version 1's doc-parallel byte figure was
  taken at B=50 and multiplied by ten while the caption called it measured.
- `nsys_scatter_kern_sum.txt`: the independent profiler cross-check of the
  component decomposition.

## Superseded

Anything older reflects pre-audit numbers, including the 0.37% and 62.6%
bandwidth figures that were analytic byte counts wrong by 42x and 1.56x in
opposite directions, the Seismic runs at heap_factor=10.0 (outside the
parameter's documented domain), and the 0.999 recall figures that came from
truncating queries to 64 terms. The full list of corrections is in the arXiv v2
"Comments" field on the paper's abstract page.

## Rule going forward

Every published number comes from `src/remeasure_2026_09.py` or from a script
named in the paper, run against one build. Latency is CUDA-event timed; counters
come only from Nsight Compute, whose wall times are inflated ~1.56x by kernel
replay and are never read as latency.
