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

(See "Outstanding" below — the benchmark run is pending the in-flight
rebuild triggered by `linalg_impl.h` being a transitively-included
header. Numbers will be appended once `bench_cublaslt.py --driver`
completes on GPU 1. Expectation per scoping doc:
1024^3 ~1.4x, 4096^3 ~1.7x, 8192^3 ~1.5x on Blackwell.)

## Outstanding

1. **Benchmark.** Run `CUDA_VISIBLE_DEVICES=1 python bench_cublaslt.py
   --driver` after the rebuild lands. Append the speedup table here.
2. **Numerics parity test.** `pytest -q
   tests/python/gpu/test_cublaslt_gemm.py` should pass at TF32
   tolerance (5e-3 rel). Reruns required only if a future PR widens
   the compute-type set.
3. **Smoke.** `pytest tests/python/dnnl/subgraphs/test_fc_subgraph.py
   -q` should be unaffected (no CUBLASLT call sites in oneDNN paths)
   and must remain green.

## Risks observed

- `CUBLAS_COMPUTE_32F_FAST_TF32` will produce ~1e-3 rel-error drift vs
  pure FP32. The parity test tolerance accommodates this. A future
  PR-B should respect `MXNET_CUDA_ALLOW_TENSOR_CORE=0` by switching to
  `CUBLAS_COMPUTE_32F` so user requests for strict FP32 are honored.
- Per-call descriptor create/destroy is non-trivial on tiny GEMMs.
  If profiling shows it's hot, a future PR can cache the layout/desc
  triples alongside the algo in `Entry`.
