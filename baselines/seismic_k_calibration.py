"""Calibrate our single-core Seismic figure against the published one by sweeping k.

Why this exists. The paper reports Seismic at 7.071 ms/query on one pinned core
at k=1000, and Seismic's own publications report 187 to 531 us. That is a 17x
gap. The paper explains it (their figures are k=10, ours are k=1000, and at
k=1000 the heap minimum collapses so the block-skip predicate stops firing) but
never measures the k=10 point. A reviewer who knows Seismic will assume our core
pinning is broken until they see a single-core k=10 number that lands in or near
the published range. This measures it.

The claim under test: at k=10, single core, inside the documented parameter
domain, Seismic lands near its published 187 to 531 us. If it does, the pinning
is clean and the k=1000 figure is a real property of large-k search. If it lands
at milliseconds instead, our pinning or our machine is the explanation and the
paper's CPU comparison needs rewriting.

Method notes that matter for this specific library:
  - `num_threads` is inert through the Python API. `pyseismic-lsr` builds a Rayon
    pool and drops it without `.install()`, so `par_bridge` uses the global pool
    sized to all logical CPUs. Single-core therefore requires
    RAYON_NUM_THREADS=1 (which does control the global pool) and `taskset`.
    Both are set by the launcher. This script verifies the pinning held rather
    than assuming it: it reports CPU time next to wall time, and a ratio far
    above 1.0 means the pin leaked and the row is void.
  - heap_factor must lie in (0,1); the library does not validate it. We stay at
    0.7 (the paper's setting) and also report 0.9, nearer Seismic's own tuning.
  - Latency is reported as batch wall time over query count, matching how the
    published figures and the paper's existing table are computed.
"""

import json
import os
import resource
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("GPUSPARSE_DATA_SEISMIC", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/seismic"))
os.environ.setdefault("SEISMIC_DATA", DATA)
sys.path.insert(0, f"{REPO}/baselines")

import seismic  # noqa: E402
from run_seismic_multithread import load_queries, load_qrels, evaluate_results  # noqa: E402

OUT = f"{REPO}/final_results/remeasure_2026_09/seismic_k_calibration.json"


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    idx_path = os.path.join(DATA, "seismic_index.bin.index.seismic")
    print(f"loading {idx_path}", flush=True)
    t0 = time.time()
    index = seismic.SeismicIndex.load(idx_path)
    print(f"  {index.len:,} docs in {time.time() - t0:.1f}s", flush=True)

    qids, qcomp, qvals = load_queries()
    qrels = load_qrels()
    st = seismic.get_seismic_string()
    qids_arr = np.array(qids, dtype=st)
    nq = len(qids)
    print(f"  {nq} queries, {len(qrels)} with qrels", flush=True)
    print(f"  RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')!r} "
          f"affinity={sorted(os.sched_getaffinity(0))}", flush=True)

    res = {
        "_why": "settle the k=10 single-core calibration row; FUTURE_WORK item 6",
        "index_docs": int(index.len),
        "n_queries": nq,
        "n_queries_with_qrels": len(qrels),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "published_reference_us": {"low": 187, "high": 531, "k": 10,
                                   "source": "bruch2024seismic reported range"},
        "paper_row_to_calibrate": {"k": 1000, "ms_per_query": 7.071, "cores": 1,
                                   "heap_factor": 0.7, "query_cut": 20},
        "runs": [],
    }

    # Warm the index pages so the first timed row is not paying page faults.
    _ = index.batch_search(qids_arr[:200], qcomp[:200], qvals[:200],
                           k=10, query_cut=20, heap_factor=0.7, num_threads=1)

    configs = []
    for hf, qc in ((0.7, 20), (0.9, 10)):
        for k in (10, 100, 1000):
            configs.append((k, hf, qc))

    print(f"\n{'k':>5} {'heap_f':>7} {'q_cut':>6} {'wall(s)':>8} {'ms/query':>9} "
          f"{'us/query':>9} {'cpu/wall':>9} {'MRR@10':>8} {'R@k':>7}", flush=True)
    print("-" * 82, flush=True)

    for k, hf, qc in configs:
        c0, w0 = cpu_seconds(), time.time()
        out = index.batch_search(qids_arr, qcomp, qvals,
                                 k=k, query_cut=qc, heap_factor=hf, num_threads=1)
        wall = time.time() - w0
        cpu = cpu_seconds() - c0
        m = evaluate_results(out, qids, qrels, k_mrr=min(10, k), k_recall=k)
        # evaluate_results names its recall key 'recall@1000' whatever k_recall
        # was, so relabel it here to the cutoff actually used.
        recall_at_k = m["recall@1000"]
        row = {
            "k": k, "heap_factor": hf, "query_cut": qc,
            "wall_s": wall, "ms_per_query": wall / nq * 1000,
            "us_per_query": wall / nq * 1e6,
            "cpu_s": cpu, "cpu_over_wall": cpu / wall if wall else None,
            "qps": nq / wall,
            "mrr@10": m["mrr@10"], f"recall@{k}": recall_at_k,
            "num_evaluated": m["num_evaluated"],
            "pinning_clean": bool(cpu / wall < 1.35) if wall else None,
        }
        res["runs"].append(row)
        print(f"{k:>5} {hf:>7.1f} {qc:>6} {wall:>8.2f} {wall / nq * 1000:>9.3f} "
              f"{wall / nq * 1e6:>9.1f} {row['cpu_over_wall']:>9.2f} "
              f"{m['mrr@10']:>8.4f} {recall_at_k:>7.4f}", flush=True)
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)

    # Verdict. The question is whether the k=10 single-core point is consistent
    # with the published range, which is what tells a reader our pinning is fine.
    k10 = [r for r in res["runs"] if r["k"] == 10]
    k1000 = [r for r in res["runs"] if r["k"] == 1000]
    best10 = min(k10, key=lambda r: r["us_per_query"])
    worst1000 = max(k1000, key=lambda r: r["ms_per_query"])
    leaked = [r for r in res["runs"] if r["pinning_clean"] is False]
    res["verdict"] = {
        "best_k10_us_per_query": best10["us_per_query"],
        "in_published_range": 187.0 <= best10["us_per_query"] <= 531.0,
        "within_2x_of_published_high": best10["us_per_query"] <= 2 * 531.0,
        "k1000_over_k10_ratio": worst1000["ms_per_query"] * 1000 / best10["us_per_query"],
        "rows_with_leaked_pinning": len(leaked),
        "reading": (
            "k=10 near the published range confirms the pinning is clean and the "
            "7.07 ms k=1000 figure is a property of large-k search, not of our setup."
        ),
    }
    print("\n### verdict ###", flush=True)
    print(f"  best k=10 single core: {best10['us_per_query']:.1f} us/query "
          f"(published range 187 to 531)", flush=True)
    print(f"  in published range: {res['verdict']['in_published_range']}, "
          f"within 2x of high end: {res['verdict']['within_2x_of_published_high']}", flush=True)
    print(f"  k=1000 / k=10 ratio: {res['verdict']['k1000_over_k10_ratio']:.1f}x", flush=True)
    print(f"  rows where pinning leaked: {len(leaked)}", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
