"""Verify + benchmark the two new kernels: chunked GPUSparse scorer, CUDA PQ kernel."""
import json, os, sys, time
import numpy as np, scipy.sparse as sp, torch

GS=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FM=os.environ.get("TILEMAXSIM_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"))
sys.path.insert(0,f"{GS}/src"); sys.path.insert(0,GS); sys.path.insert(0,f"{FM}/src")
OUT=os.path.join(os.environ.get("GPUSPARSE_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")), "results.json")
res={}

def t_ms(fn, warmup=5, iters=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

# ============================ 1. chunked GPUSparse scorer
print("="*72); print("1. CHUNKED SCORER: O(B*chunk) vs O(B*N) memory"); print("="*72, flush=True)
from triton_kernel import triton_fused_score
from triton_kernel_chunked import triton_chunked_score, build_chunk_boundaries
from run_correctness_verification import build_gpu_index_from_csr, prepare_queries_from_meta
rows={}
for N in (100000, 1000000):
    csr = sp.load_npz(f"{GS}/data_cache/msmarco_splade_csr_{N}.npz").tocsr().astype(np.float32)
    meta = torch.load(f"{GS}/data_cache/msmarco_splade_csr_{N}_meta.pt", map_location="cpu", weights_only=False)
    qi,qs,_ = prepare_queries_from_meta(meta, max_terms=128, device="cuda:0")
    B=500; qi,qs = qi[:B].contiguous(), qs[:B].contiguous()
    idx = build_gpu_index_from_csr(csr, "cuda:0")
    t0=time.time(); cb = build_chunk_boundaries(idx, 131072); bt=time.time()-t0
    print(f"\n-- N={N:,} B={B} k=1000 (chunk boundary build: {bt:.1f}s, {cb.numel()*8/2**20:.1f} MiB) --", flush=True)

    # correctness
    s_mono, i_mono = triton_fused_score(idx, qi, qs, top_k=1000)
    s_chk,  i_chk  = triton_chunked_score(idx, qi, qs, top_k=1000, chunk_bounds=cb)
    torch.cuda.synchronize()
    ov = float(np.mean([len(set(i_mono[r,:1000].tolist()) & set(i_chk[r,:1000].tolist()))/1000
                        for r in range(B)]))
    serr = float((s_mono.sort(dim=1,descending=True).values - s_chk.sort(dim=1,descending=True).values).abs().max())
    print(f"   top-1000 overlap vs monolithic: {ov:.6f} | max score diff: {serr:.3e}", flush=True)

    # memory + latency
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t_m = t_ms(lambda: triton_fused_score(idx, qi, qs, top_k=1000))
    m_m = torch.cuda.max_memory_allocated()/2**30
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t_c = t_ms(lambda: triton_chunked_score(idx, qi, qs, top_k=1000, chunk_bounds=cb))
    m_c = torch.cuda.max_memory_allocated()/2**30
    print(f"   monolithic: {t_m:8.2f} ms   peak {m_m:6.2f} GiB")
    print(f"   chunked   : {t_c:8.2f} ms   peak {m_c:6.2f} GiB")
    print(f"   -> memory {m_m/max(m_c,1e-9):5.2f}x lower, latency {t_c/t_m:5.2f}x", flush=True)
    rows[f"N{N}"]={"overlap":ov,"max_score_diff":serr,"mono_ms":t_m,"chunked_ms":t_c,
                   "mono_peak_gib":m_m,"chunked_peak_gib":m_c,
                   "memory_reduction":m_m/max(m_c,1e-9),"latency_ratio":t_c/t_m,
                   "boundary_build_s":bt,"boundary_mib":cb.numel()*8/2**20}
    del idx, cb; torch.cuda.empty_cache()
res["chunked_scorer"]=rows

# ============================ 2. CUDA PQ kernel
print("\n"+"="*72); print("2. CUDA PQ KERNEL: shared-memory staged table"); print("="*72, flush=True)
from torch.utils.cpp_extension import load
ext = load(name="pqsim_cuda", sources=[f"{FM}/src/pqsim_cuda.cu"],
           extra_cuda_cflags=["-O3","--use_fast_math"], verbose=False)
from flash_pqsim_kernel import FlashPQSim
Nq,Nd,M,K,dsub = 32,128,16,256,8
cb_ = torch.randn(M,K,dsub, device="cuda", dtype=torch.float16)
pq = FlashPQSim(cb_)
Q = torch.randn(Nq, M*dsub, dtype=torch.float16, device="cuda")
table = pq.build_distance_table(Q).contiguous()
pqrows={}
for B in (20000, 100000):
    codes = torch.randint(0,K,(B,Nd,M),device="cuda",dtype=torch.uint8)
    ref_old = pq.score_batch(Q, codes, multiquery=False)
    assert float((ref_old-pq.score_batch_mq(Q, codes)).abs().max()) < 1e-2, \
        'Triton published and MQ paths disagree; fix before trusting CUDA deltas'
    ref_mq  = pq.score_batch_mq(Q, codes)
    torch.cuda.synchronize()
    print(f"\n-- B={B} --", flush=True)
    best=None
    for tag, fn in (("per-doc", ext.pqsim_smem), ("tiled", ext.pqsim_smem_tiled)):
        for bq in (1,2,4,8):
            try:
                got = fn(table, codes, bq); torch.cuda.synchronize()
                err = float((got-ref_mq).abs().max())
                tt = t_ms(lambda fn=fn, bq=bq: fn(table, codes, bq))
                smem = bq*M*K*4/1024
                print(f"   CUDA {tag:7s} BQ={bq}: {tt:8.3f} ms  smem {smem:6.0f} KiB  "
                      f"max_err={err:.3e}", flush=True)
                if best is None or tt<best[1]: best=(f"{tag}/BQ={bq}",tt,err)
            except Exception as ex:
                print(f"   CUDA {tag:7s} BQ={bq}: {type(ex).__name__}: {str(ex)[:60]}", flush=True)
    t_old = t_ms(lambda: pq.score_batch(Q, codes, multiquery=False))
    t_mq  = t_ms(lambda: pq.score_batch_mq(Q, codes))
    print(f"   Triton published : {t_old:8.3f} ms")
    print(f"   Triton MQ (BQ=2) : {t_mq:8.3f} ms")
    if best:
        print(f"   CUDA best ({best[0]}): {best[1]:8.3f} ms  -> {t_old/best[1]:.2f}x vs published, "
              f"{t_mq/best[1]:.2f}x vs Triton-MQ", flush=True)
        pqrows[f"B{B}"]={"cuda_best_variant":best[0],"cuda_ms":best[1],"cuda_max_err":best[2],
                         "triton_published_ms":t_old,"triton_mq_ms":t_mq,
                         "speedup_vs_published":t_old/best[1],"speedup_vs_triton_mq":t_mq/best[1]}
    del codes; torch.cuda.empty_cache()
res["cuda_pq"]=pqrows
json.dump(res, open(OUT,"w"), indent=2); print(f"\nwrote {OUT}", flush=True)
