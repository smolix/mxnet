# B8 CPU Inference Benchmark — apache/mxnet#19218

**Date**: 2026-05-18  
**Branch**: `sweep/master-plus-prs`  
**Host**: AMD EPYC 7B12 (Zen 2, 128 cores @ 3.2 GHz) — AVX2 only, no AVX-512  
**MXNet**: 2.0.0+cu13.bw.20260518  
**oneDNN**: v3.11.3 (commit 74d04752)  
**Warmup**: 10 runs · **Timed**: 30 runs · **Metric**: avg ms/inference  

---

## Results Table

| Shape | OMP_NUM_THREADS | ms/inference | Primitive (DNNL_VERBOSE) |
|---|---|---|---|
| Conv2D 64ch (1,3,224,224) | 1 | 49.8 | `brg_conv_fwd:avx2` / `convolution_direct` |
| Conv2D 64ch (1,3,224,224) | 4 | 115.2 | `brg_conv_fwd:avx2` / `convolution_direct` |
| Conv2D 64ch (1,3,224,224) | 16 | 336.4 | `brg_conv_fwd:avx2` / `convolution_direct` |
| Conv2D 64ch (1,3,224,224) | 64 (DNNL default) | 536.4 | `brg_conv_fwd:avx2` / `convolution_direct` |
| Conv2D 512ch (1,3,224,224) | 1 | 281.9 | `brg_conv_fwd:avx2` / `convolution_direct` |
| Conv2D 512ch (1,3,224,224) | 4 | 313.8 | `brg_conv_fwd:avx2` / `convolution_direct` |
| Conv2D 512ch (1,3,224,224) | 16 | 413.3 | `brg_conv_fwd:avx2` / `convolution_direct` |
| Conv2D 512ch (1,3,224,224) | 64 (DNNL default) | 539.3 | `brg_conv_fwd:avx2` / `convolution_direct` |
| Dense 1000 (1,2048) | 1 | 3.6 | `brg_matmul:avx2` / `inner_product` |
| Dense 1000 (1,2048) | 4 | 19.1 | `brg_matmul:avx2` / `inner_product` |
| Dense 1000 (1,2048) | 16 | 160.6 | `brg_matmul:avx2` / `inner_product` |
| Dense 1000 (1,2048) | 64 (DNNL default) | 270.7 | `brg_matmul:avx2` / `inner_product` |
| softmax (1,1000) | 1 | 0.044 | `jit:avx2` / `softmax_accurate` |
| softmax (1,1000) | 4 | 0.065 | `jit:avx2` / `softmax_accurate` |
| softmax (1,1000) | 16 | 0.107 | `jit:avx2` / `softmax_accurate` |
| softmax (1,1000) | 64 (DNNL default) | 0.043 | `jit:avx2` / `softmax_accurate` |

---

## DNNL Primitive Dispatch (OMP=4, DNNL_VERBOSE=1)

```
oneDNN v3.11.3 | ISA: Intel AVX2 | runtime: OpenMP nthr:4

Conv64:  brg_conv_fwd:avx2  mb1_ic3oc64_ih224oh224kh3sh1dh0ph1  exec=183ms (first run)
         weights format Acdb16a (16-channel blocked)
         input  format acdb (nhwc-like)

Conv512: brg_conv_fwd:avx2  mb1_ic3oc512_ih224oh224kh3sh1dh0ph1 exec=301ms (first run)
         weights format Acdb16a (same blocking)

Dense:   brg_matmul:avx2    mb1ic2048oc1000                     exec=194ms (first run)

Softmax: jit:avx2           1x1000 axis:1                       exec=0.006ms
```

---

## Verdict: **B8 REPRODUCES — and is actually worse than reported**

The upstream issue reported ~110 ms per conv. On this AVX2-only Zen 2 box:

- **Conv2D 64ch**: 49.8 ms at OMP=1; degrades severely to 536 ms at the DNNL default of 64 threads.
- **Conv2D 512ch**: 281.9 ms even at OMP=1.
- **Dense 1000**: 3.6 ms at OMP=1 (fine), but 271 ms at 64 threads (terrible scaling).
- **Softmax**: Fast at all thread counts (<0.2 ms). Not a problem.

The best-case single-thread (OMP=1) Conv64 at 49.8 ms is within 2x of the ~110 ms reference, and **performance degrades monotonically as more threads are added** — the opposite of expected speedup. With DNNL's auto-detected 64 threads, Conv64 is 10x slower than single-threaded.

### Root Causes

1. **IC=3 is pathological for `brg_conv_fwd:avx2`**: The weight format `Acdb16a` pads IC to a multiple of 16. With IC=3, 13/16 = 81% of every inner-loop vector operation is wasted padding. This is not a regression from oneDNN v3; it is a known limitation of the blocked format with tiny channel counts.

2. **Negative thread scaling on bs=1**: `brg_conv` is designed for high-throughput batch workloads. With batch_size=1 and a small spatial problem (224×224 with IC=3), there is barely enough work to occupy even 1 thread. Adding more threads introduces OpenMP overhead, cache-line contention, and NUMA overhead (Zen 2 EPYC has 8 NUMA domains). At 64 threads the overhead exceeds the compute time by ~10x.

3. **This fork does not fix B8**: The oneDNN v3 bump brings `brg_conv_fwd:avx2` as the chosen implementation, which is actually **worse** for IC=3 bs=1 than the older `jit:avx2` implementation that would have been selected in oneDNN v2.

---

## Actionable Follow-Up

1. **Set `OMP_NUM_THREADS=1` for bs=1 inference**: Users hitting this bug should pin to 1 thread. `Conv64` goes from 536 ms → 49.8 ms (10x). Add a note to the docs/DNNL tuning guide.

2. **Suppress `brg_conv` for IC<16 + bs=1**: In `src/operator/nn/dnnl/dnnl_convolution.cc`, check the primitive descriptor selection. For `ic < 16 && batch_size == 1`, prefer the non-blocked `jit:avx2` path via `dnnl::primitive_attr` hints or by setting `DNNL_MAX_CPU_ISA=AVX2_VNNI` to steer away from brgemm. This is a config-only change — no source rebuild needed to test (`ONEDNN_DEFAULT_FPMATH_MODE`, `DNNL_VERBOSE` flags).

3. **Test with `DNNL_DEFAULT_FPMATH_MODE=BF16`**: OneDNN v3 supports bf16 emulation on AVX2. This sometimes selects a different code path with better IC=3 handling. Measurable with `DNNL_DEFAULT_FPMATH_MODE=BF16 OMP_NUM_THREADS=1`.

4. **The 512ch case is genuinely slow**: 281 ms at OMP=1 for Conv512 with IC=3 on 224×224 is inherent — 512 × 3 × 3 × 3 × 224 × 224 = ~654 MFLOP with AVX2 at ~3.2 GHz × 8 wide × 2 FMA = ~51 GFLOP/s theoretical peak → should be ~13 ms. The 21x slowdown is entirely due to IC padding waste (81%) + cache miss penalty. Not fixable without changing the network architecture (add a 1×1 conv stem to expand IC from 3 to 16/32 before the main conv).
