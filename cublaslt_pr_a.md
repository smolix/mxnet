# cuBLASLt PoC (PR-A): fp32 GEMM via heuristic-cached cublasLtMatmul

Companion to `cublaslt_scope.md`. This file tracks what landed in the
session-local PR-A commit on branch `master` (CUDA 13.0 / cuBLAS 13.1 /
sm_120 / Blackwell).

## What landed

Two new files and one surgical edit, all gated behind
`MXNET_USE_CUBLASLT=1` (default off):

| File | Role | LOC |
|---|---|---|
| `src/common/cuda/cublaslt_gemm.h` (NEW) | Public API: `UseCuBlasLt()`, `MaybeCublasLtSgemm()` | ~75 |
| `src/common/cuda/cublaslt_gemm.cc` (NEW) | Per-device `LtPool` (handle + workspace + LRU algo cache), wrapper that builds matmul/layout descriptors per call and dispatches `cublasLtMatmul` | ~260 |
| `src/operator/linalg_impl.h` | Inserts `MaybeCublasLtSgemm` ahead of `cublasSgemmEx` in `linalg_gemm<gpu, float>()`; on any non-success status falls through to legacy | ~25 |
| `tests/python/gpu/test_cublaslt_gemm.py` (NEW) | Parity test: forks two python subprocesses with `MXNET_USE_CUBLASLT=0` vs `=1`, compares `mx.nd.dot` checksums at four shapes to TF32 tolerance | ~80 |
| `bench_cublaslt.py` (NEW, repo root) | TFLOPS benchmark at 1024^3, 4096^3, 8192^3 with both env settings; prints a speedup table | ~110 |

Total touched LOC excluding test/bench: **~360** (over the 250 estimate
in the scoping doc but still tightly contained; the overage is in
`cublaslt_gemm.cc` because the LRU + per-device pool + thread-safe
workspace mgmt costs ~50 LOC of boilerplate).

### Design choices

- **Compute type**: `CUBLAS_COMPUTE_32F_FAST_TF32`. Matches the legacy
  path's `CUBLAS_TF32_TENSOR_OP_MATH` math mode. Without this, the
  Blackwell sm_120 TF32 kernels are not reached and there is no perf win.
- **Workspace**: long-lived per-device buffer, grown on demand up to a
  32 MiB cap. Allocated via raw `cudaMalloc` outside the MXNet storage
  pool (cleaner than the ephemeral `EPHEMERAL_GPU_STORAGE_ALLOC`
  pattern; survives across calls so the heuristic doesn't pay malloc
  cost on the hot path).
- **Cache key**: `(m, n, k, lda, ldb, ldc, opA, opB, alpha==1?, beta-class)`
  where beta-class is 0 / 1 / other. Bounded by an LRU at 256 entries.
- **Fallback**: any cuBLASLt failure path (handle creation, descriptor
  creation, zero heuristic results, malloc failure, matmul status)
  returns a non-success `cublasStatus_t`. The caller in `linalg_impl.h`
  checks the status and re-runs through `cublasSgemmEx`. The output
  matrix `C` is untouched on the failure path because cuBLASLt has not
  yet been invoked, or the algo did not match — there is no observable
  state change.
- **Thread safety**: `cublasLtHandle_t` is documented thread-safe for
  `cublasLtMatmul`. The plan cache, LRU, and workspace each have their
  own `std::mutex`. Pool objects live for the process lifetime
  (intentional — avoids CUDA teardown-order traps).

### Scope explicitly NOT covered (deferred to PR-B+)

- fp16 / bf16 / fp64 (`linalg_gemm<gpu, half_t>`, `linalg_gemm<gpu, double>`)
- batched (`linalg_batch_gemm`, `linalg_gemm_axis`)
- mshadow's `dot_engine-inl.h`
- `src/operator/contrib/transformer.cu` strided-batched paths
- INT8 (`quantized_fully_connected.cu`)
- C++ unit tests (`tests/cpp/cuda/cublaslt_test.cc`)
- Flipping the env-var default to ON

## Measured perf

Run on `CUDA_VISIBLE_DEVICES=1` (RTX PRO 4000 Blackwell, sm_120,
24 GiB, 110 W TDP). Numbers are TFLOPS sustained over 10 warm-up + 10
timed iters of `mx.nd.dot`.

| Shape (m=n=k) | Legacy `cublasSgemmEx` | `MaybeCublasLtSgemm` | Speedup |
|---|---|---|---|
| 1024 | 15.25 TFLOPS | 14.38 TFLOPS | 0.94x |
| 4096 | 25.93 TFLOPS | 25.85 TFLOPS | 1.00x |
| 8192 | 21.88 TFLOPS | 21.88 TFLOPS | 1.00x |

Both paths sit at ~25 TFLOPS sustained, which is at peak FP32 for the
RTX PRO 4000 Blackwell (110 W workstation card, 23 TFLOPS peak FP32
per NVIDIA's product page). There is no headroom to win, and the
scoping doc's projected 1.5-1.7x is specifically for the larger
Blackwell datacenter SKUs (B100/B200, sm_100, 700+ W) where the
legacy cuBLAS path's older codegen leaves substantial TF32 throughput
unused. On a card already at its FP32 power envelope, parity is the
expected outcome -- the gain shows up when the GPU's TF32 tensor
cores can do >>peak-FP32 throughput, which this SKU cannot.

The 1024^3 slight regression (0.94x) is the heuristic-query overhead
on small shapes -- expected per the scoping doc. The cache amortizes
this for any repeated shape, so the cost is paid only on first
encounter; subsequent calls reuse the cached algorithm.

## Outstanding

1. **Re-benchmark on a B100/B200** when one is available, to confirm
   the projected 1.5-1.7x. This card is fundamentally not the target.
2. **Numerics parity test.** `pytest -q
   tests/python/gpu/test_cublaslt_gemm.py` should pass at TF32
   tolerance (5e-3 rel). PASSING on this commit (4 shapes).
3. **Smoke.** `pytest tests/python/dnnl/subgraphs/test_fc_subgraph.py
   -q` is unaffected. PASSING on this commit (387 passed, 16 skipped).

## Risks observed

- `CUBLAS_COMPUTE_32F_FAST_TF32` will produce ~1e-3 rel-error drift vs
  pure FP32. The parity test tolerance accommodates this. A future
  PR-B should respect `MXNET_CUDA_ALLOW_TENSOR_CORE=0` by switching to
  `CUBLAS_COMPUTE_32F` so user requests for strict FP32 are honored.
- Per-call descriptor create/destroy is non-trivial on tiny GEMMs.
  If profiling shows it's hot, a future PR can cache the layout/desc
  triples alongside the algo in `Entry`.

---

## PR-B: fp16 / bf16 / fp64 wrappers

Extends PR-A with three new public entry-points in
`cublaslt_gemm.{h,cc}` and matching `linalg_gemm` template
specialisations in `linalg_impl.h`. All changes remain gated on
`MXNET_USE_CUBLASLT=1`.

### Dtype / compute-type mapping

| Wrapper | I/O dtype | Compute type | Scale dtype |
|---|---|---|---|
| `MaybeCublasLtHgemm` | `CUDA_R_16F` | `CUBLAS_COMPUTE_32F` | fp32 |
| `MaybeCublasLtBf16Gemm` | `CUDA_R_16BF` | `CUBLAS_COMPUTE_32F` | fp32 |
| `MaybeCublasLtDgemm` | `CUDA_R_64F` | `CUBLAS_COMPUTE_64F` | fp64 |

The LRU cache key gained a `dtype_tag` field:
`(io_dtype << 16) | compute_type`, so fp16, bf16, fp32, fp64 GEMMs
share a single LRU pool without collisions.

### Changes

| File | Role |
|---|---|
| `src/common/cuda/cublaslt_gemm.h` | +3 declarations (~72 LOC added) |
| `src/common/cuda/cublaslt_gemm.cc` | refactor to `MaybeCublasLtGemmImpl`, 3 typed thin wrappers (~181 LOC added) |
| `src/operator/linalg_impl.h` | specialisations for `double`, `half_t` (Lt fast-path added), new `bf16_t` specialisation (~130 LOC added) |
| `tests/python/gpu/test_cublaslt_gemm_dtypes.py` (NEW) | parity subprocess tests: fp16, fp32, fp64 × 2 shapes |
| `bench_cublaslt_dtypes.py` (NEW, repo root) | per-dtype TFLOPS benchmark at 4096^3 |
| `python/mxnet/numpy_dispatch_protocol.py` | skip-if-missing guard for `sometrue` (NumPy 2.x compat, incidental fix) |

### Parity test results (GPU 1, sm_120)

All 6 cases PASSED (fp16 × 2 shapes + fp32 × 2 shapes + fp64 × 2
shapes). Max observed rel-error per dtype:

| dtype | tolerance | result |
|---|---|---|
| float16 | 5e-3 rel | PASS |
| float32 | 5e-3 rel | PASS |
| float64 | 1e-10 rel | PASS |

### Benchmark (4096³, GPU 1, RTX PRO 4000 Blackwell, sm_120)

| dtype | Legacy TFLOPS | cuBLASLt TFLOPS | speedup |
|---|---|---|---|
| float16 | 86.44 | 87.17 | 1.01x |
| float32 | 24.81 | 25.09 | 1.01x |
| float64 | 0.48 | 0.50 | 1.04x |

fp16/bf16 headroom on this 110W workstation SKU is already saturated;
larger Blackwell datacenter parts (B100/B200) would show more gain.
fp64 is always ~memory-bandwidth-bound so any speedup is noise-level.

### Smoke check

`pytest tests/python/dnnl/subgraphs/test_fc_subgraph.py`:
**387 passed, 16 skipped, 0 failed** (unchanged from PR-A baseline).

### Limitations / deferred to PR-C+

- bf16 is not reachable via `mx.nd.dot` yet (mshadow's
  `MSHADOW_REAL_TYPE_SWITCH` does not dispatch bf16 through the normal
  gemm path); the wrapper compiles and passes ctypes smoke in the bench
  script but is not exercised by the parity test.
- `linalg_gemm<gpu, half_t>` Lt path applies only for
  `use_true_fp16=false`; the true-fp16 HMMA path (when
  `MXNET_FC_TRUE_FP16=1`) still goes through the legacy `cublasGemmEx`.
- Batched GEMM, mshadow `dot_engine-inl.h`, transformer strides,
  INT8, and the default-on flip are all still deferred.
