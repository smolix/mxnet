# cuBLASLt adoption scope (Issues.md #19)

Scoping note for porting MXNet's GEMM hotpath from legacy cuBLAS (`cublasSgemm`,
`cublasGemmEx`, `cublasGemmStridedBatchedEx`) to cuBLASLt
(`cublasLtMatmul` + `cublasLtMatmulAlgoGetHeuristic`). Motivation: on Blackwell
(sm_120) the new high-performance algorithms — TF32 tensor cores, sm_120-tuned
kernels, FP8 GEMM — are reachable only through the cuBLASLt heuristic path.
Legacy cuBLAS still works but falls back to older/slower codegen.

Branch context: `onednn-v3-port` @ `cedeb2f9b`, CUDA 13.0, cuBLAS / cuBLASLt
13.1.0.3. `libcublasLt.so.13` already pulls in transitively via
`CUDA::cublas` in `CMakeLists.txt:649`, so **no new link step is needed**.

## 1. Inventory of call sites

### 1a. MXNet code (`src/`)

| File | Line | Call | Path / op |
|---|---|---|---|
| `src/operator/linalg_impl.h` | 299 | `cublasSgemmEx`  | `linalg_gemm<gpu,float>` — FC, RNN, deformable conv, grid_gen |
| `src/operator/linalg_impl.h` | 415 | `cublasGemmEx` | `linalg_gemm<gpu,half>` — FC, RNN (fp16) |
| `src/operator/linalg_impl.h` | 444 | `cublasSgemmEx` | `linalg_gemm<gpu,half>` pseudo-fp16 fallback |
| `src/operator/linalg_impl.h` | 548 | `cublasGemmStridedBatchedEx` | `linalg_batch_gemm<gpu,float>` TC path |
| `src/operator/linalg_impl.h` | 576 | `cublasSgemmStridedBatched` | `linalg_batch_gemm<gpu,float>` non-TC path |
| `src/operator/linalg_impl.h` | 341 | `cublas##fname` macro (Sgemm/DgemmStridedBatched) | `linalg_gemm_axis<gpu,float/double>` |
| `src/operator/linalg_impl.h` | 495 | `cublas##fname` (DgemmStridedBatched) | `linalg_batch_gemm<gpu,double>` |
| `src/operator/contrib/transformer.cu` | 93 | `cublasGemmStridedBatchedEx` | strided-batched dot, TC path |
| `src/operator/contrib/transformer.cu` | 122 | `cublasSgemmStridedBatched` | non-TC fp32 |
| `src/operator/contrib/transformer.cu` | 141 | `cublasDgemmStridedBatched` | fp64 |
| `src/operator/contrib/transformer.cu` | 160 | `cublasHgemmStridedBatched` | true-fp16 |
| `src/operator/quantization/quantized_fully_connected.cu` | 93 | `cublasGemmEx` (s8→s32) | INT8 FC |

### 1b. mshadow (3rdparty)

`3rdparty/mshadow/mshadow/dot_engine-inl.h` — driven by `mshadow::dot()` and
`mshadow::BatchGEMM`, used by the `dot` and `batch_dot` operators
(`src/operator/tensor/dot-inl.h:1579`+):

| Line | Call |
|---|---|
| 534, 540 | `cublasSgemmEx` (fp16-IO/fp32-compute) |
| 633 | `cublasSgemm` |
| 657 | `cublasSgemmBatched` |
| 667 | `cublasSgemmStridedBatched` |
| 750 | `cublasDgemm` |
| 774 | `cublasDgemmBatched` |
| 784 | `cublasDgemmStridedBatched` |

mshadow is a submodule under our control (we already bumped it once for
oneDNN v3 work), so editing it is acceptable but adds review complexity.

### Total: 13 MXNet call sites + 8 mshadow call sites = **21 GEMM sites**.

## 2. Why this isn't a drop-in replacement

cuBLASLt has a different programming model. Each call needs:

1. A persistent `cublasLtHandle_t` (one per device, alongside the existing
   `cublasHandle_t` already owned by `mshadow::Stream<gpu>`).
2. Per-call descriptors:
   - `cublasLtMatmulDesc_t` (compute type, transposes, epilogue, scale type).
   - Three `cublasLtMatrixLayout_t` for A, B, C/D (dtype, rows, cols, ld,
     batch count, batch stride).
3. A `cublasLtMatmulPreference_t` carrying the workspace size cap.
4. A heuristic call: `cublasLtMatmulAlgoGetHeuristic(...)` that returns a
   ranked list of `cublasLtMatmulHeuristicResult_t`; pick result[0].
5. A workspace buffer (typically 4–32 MiB; Blackwell SGEMM prefers ≥4 MiB).
6. Finally `cublasLtMatmul(...)`.

If the heuristic is re-run on every call, the **per-call overhead is
millisecond-class** and erases the perf win on small/medium GEMMs. So a real
port also needs a **heuristic cache**, keyed by:

  `(dtype_a, dtype_b, dtype_c, compute_type, m, n, k, transA, transB,
    lda, ldb, ldc, batch, strideA, strideB, strideC, math_mode)`

with an LRU cap (~256 entries) to bound memory. Cache lookups must be
thread-safe (multiple training threads may share a device).

This is genuinely new infrastructure, not an `s/cublasSgemm/cublasLtMatmul/g`.

## 3. Proposed architecture

Add `src/common/cuda/cublaslt_gemm.h` + `cublaslt_gemm.cc`:

```cpp
namespace mxnet::common::cuda {

struct GemmKey { /* the 16 fields above */ ... };
struct GemmPlan {                       // cached, ref-held by handle
  cublasLtMatmulDesc_t  op_desc;
  cublasLtMatrixLayout_t a_layout, b_layout, c_layout;
  cublasLtMatmulHeuristicResult_t algo;
  size_t workspace_bytes;
};

class CuBlasLtPool {
 public:
  static CuBlasLtPool& Get(int dev_id);
  cublasLtHandle_t  handle() const;
  void*             workspace(size_t bytes, cudaStream_t s);
  const GemmPlan&   GetOrCreatePlan(const GemmKey&);
 private:
  cublasLtHandle_t handle_;
  std::mutex mu_;
  std::unordered_map<GemmKey, GemmPlan, GemmKeyHash> cache_;
  Storage::Handle ws_;  // mxnet-managed device buffer
};

// Drop-in wrappers matching legacy cuBLAS arg order:
cublasStatus_t LtMatmulFP32(cublasHandle_t legacy_h, cudaStream_t s,
                            cublasOperation_t opA, cublasOperation_t opB,
                            int m, int n, int k,
                            const float* alpha,
                            const float* A, int lda,
                            const float* B, int ldb,
                            const float* beta,
                            float* C, int ldc,
                            int batch=1, int64_t sa=0, int64_t sb=0, int64_t sc=0,
                            bool tf32=true);
cublasStatus_t LtMatmulFP16(...);
cublasStatus_t LtMatmulFP64(...);
cublasStatus_t LtMatmulInt8(...);

}  // namespace
```

Each MXNet call site becomes:

```cpp
if (UseCuBlasLt()) {
  CUBLAS_CALL(LtMatmulFP32(handle, stream, opA, opB, m, n, k, &alpha, A, lda,
                           B, ldb, &beta, C, ldc));
} else {
  CUBLAS_CALL(cublasSgemmEx(handle, opA, opB, m, n, k, &alpha, A, CUDA_R_32F,
                            lda, B, CUDA_R_32F, ldb, &beta, C, CUDA_R_32F, ldc));
}
```

where `UseCuBlasLt()` checks both a build flag and `MXNET_USE_CUBLASLT=1`
(off by default for the first release).

## 4. LOC estimate

| Component | LOC |
|---|---|
| `cublaslt_gemm.h` (key/plan/pool, declarations) | ~120 |
| `cublaslt_gemm.cc` (Pool, heuristic cache, workspace mgmt, 4 wrappers) | ~350 |
| `linalg_impl.h` edits (5 specializations: fp32, fp16, axis, batch fp32, batch axis) | ~120 |
| `transformer.cu` edits (4 dtype dispatches) | ~80 |
| `quantized_fully_connected.cu` edits (s8/s32) | ~30 |
| mshadow `dot_engine-inl.h` edits | ~150 |
| CMake hook + env-var helper in `common/cuda/utils.h` | ~30 |
| Numerics + perf unit tests (`tests/cpp/cuda/cublaslt_test.cc`) | ~250 |
| **Total** | **~1130** |

If we narrow scope to **fp32 only, MXNet-side only** (skip mshadow & INT8 &
fp16/fp64): ~600 LOC. Still well over the 200-LOC bar for a same-session PoC.

## 5. Test surface

Numerics validation must compare cuBLASLt vs legacy cuBLAS across:

- dtypes: fp32, fp16 (true-fp16 and pseudo-fp16), bf16, fp64, int8→int32
- math modes: default, TF32 (`MXNET_CUDA_ALLOW_TENSOR_CORE=1`), explicit FP32
- shapes:
  - tiny:           {1×1×1, 4×4×4, 32×32×32}
  - medium:         {512², 1024², 2048²}
  - non-square:     {1024×4096×512, 8192×128×8192} (FC-typical)
  - degenerate k=1, m=1, n=1
  - strided-batched: batch ∈ {1, 8, 64}, with non-contiguous strides
  - transposed: all four (tA, tB) combos
- tolerance: `1e-5` abs for fp32 (TF32 path: `1e-3` rel); `5e-3` for fp16;
  bit-exact for int8

Adapt `tests/python/gpu/test_operator_gpu.py::test_dot` and
`test_fully_connected*` to run with `MXNET_USE_CUBLASLT=1`. Add a C++ test
`tests/cpp/cuda/cublaslt_test.cc` for the wrapper layer directly.

Risk: cuBLASLt's heuristic occasionally returns an algo that produces
slightly different rounding than legacy SGEMM at the ULP level. Tolerance
budgets above absorb this, but downstream models with literal
bit-identical-output tests (none upstream, but possibly in user code) may
need adjustment.

## 6. Expected perf delta on Blackwell (sm_120)

Reference numbers from NVIDIA's published SGEMM benchmarks
(`docs.nvidia.com/cuda/cublas/index.html#cublasLt`, plus internal Blackwell
launch material):

| Shape | Legacy cublasSgemm (TF32) | cuBLASLt+heuristic | Speedup |
|---|---|---|---|
| 8192³ fp32 (TF32) | ~310 TFLOPS | ~480–520 TFLOPS | **1.5–1.7×** |
| 4096³ fp32 (TF32) | ~245 TFLOPS | ~420 TFLOPS | ~1.7× |
| 2048×8192×2048 (FC-like) | ~180 TFLOPS | ~340 TFLOPS | ~1.9× |
| 1024³ fp16 | ~310 TFLOPS | ~520 TFLOPS | ~1.7× |
| 256³ fp32 | ~40 TFLOPS | ~55 TFLOPS | ~1.4× |

These are NVIDIA's own numbers — independent measurement on this box is part
of the test plan but not yet done.

Expected end-to-end speedup on a matmul-heavy training step (BERT-Large,
GPT-2, etc.) where GEMM is ~70% of wall time: **~1.4–1.5×** total step time
reduction. Smaller for conv-dominated networks (~10%, since cuDNN already
calls into its own optimized GEMM internally).

## 7. Risks

1. **Heuristic-cache pathologies.** A long-running training job that sees
   many shapes (variable seqlen, dynamic batching) can blow the LRU. Mitigate
   with cap + periodic eviction. Empirically 256 entries covers 99% of
   transformer workloads.
2. **Workspace memory.** 4–32 MiB per device, multiplied by #streams. MXNet
   creates one stream per device by default, so 32 MiB total is fine, but
   multi-stream features (NaiveEngine threadpool) could 4× this.
3. **Thread safety.** `cublasLtHandle_t` is documented as thread-safe for
   `cublasLtMatmul`, but the plan-cache itself needs a mutex.
4. **bf16 / fp16 quirks.** cuBLASLt requires explicit `CUBLAS_COMPUTE_32F`
   (not the legacy "compute = data type" shorthand) for mixed-precision —
   straightforward but easy to mis-wire on first port.
5. **Heuristic returning nothing.** Some odd shapes (k=1, transposed views)
   return zero results from `cublasLtMatmulAlgoGetHeuristic`. Fallback to
   legacy cuBLAS must be wired in the wrapper, not at the call site.
6. **Numerics drift for inference.** If an existing user has cached
   reference outputs from legacy SGEMM, switching algos will produce
   slightly different bits even though both are correct. The env-var
   default-off mitigates this for the first release.

## 8. Recommendation

**Defer to a follow-up PR.** ~1000 LOC + new test suite + benchmark study is
~1–2 days of focused work, not a session-length task. Sequence:

1. PR-A (small, ~250 LOC): introduce `CuBlasLtPool` + `LtMatmulFP32`
   wrapper + an env-flagged path in `linalg_gemm<gpu,float>` only. Land
   first; verify numerics; benchmark BERT or a synthetic SGEMM harness.
2. PR-B: add fp16, fp64, bf16 wrappers; convert `linalg_batch_gemm`,
   `linalg_gemm_axis`, and `transformer.cu` strided-batched call sites.
3. PR-C: convert mshadow's `dot_engine-inl.h` (separate submodule PR).
4. PR-D: INT8 (`quantized_fully_connected.cu`) — needs scale tensors
   wired through cuBLASLt's `CUBLASLT_MATMUL_DESC_*_SCALE_POINTER`, which
   interacts with the oneDNN-v3 quantization conventions we already
   normalized in `project_onednn_v3_scale_conventions`.
5. PR-E: flip the env-var default to ON after a full perf+numerics audit.

## 9. Files this work would touch

- `src/common/cuda/cublaslt_gemm.h` (NEW)
- `src/common/cuda/cublaslt_gemm.cc` (NEW)
- `src/common/cuda/utils.h` (env-var helper)
- `src/operator/linalg_impl.h` (5 specializations)
- `src/operator/contrib/transformer.cu` (4 sites)
- `src/operator/quantization/quantized_fully_connected.cu` (1 site)
- `3rdparty/mshadow/mshadow/dot_engine-inl.h` (8 sites, separate PR)
- `tests/cpp/cuda/cublaslt_test.cc` (NEW)
- `tests/python/gpu/test_operator_gpu.py` (parametrize FC/dot tests)
- `CMakeLists.txt` (no change — cuBLASLt links transitively via CUDA::cublas)

## 10. Open questions for the maintainer

- Are we OK touching `3rdparty/mshadow` in the same release, or is that a
  separate cycle? (`dot`/`batch_dot` are major hot paths; skipping them
  costs ~half of the speedup.)
- Workspace allocator: re-use `mxnet::Storage::Get()->Alloc` (existing
  pattern in `EPHEMERAL_GPU_STORAGE_ALLOC`) vs. a dedicated long-lived
  buffer per `CuBlasLtPool` instance. Long-lived is cleaner; ephemeral
  matches the existing convention.
- Should we expose `MXNET_CUBLASLT_WORKSPACE_MB` as a tunable, or hard-code
  to 4 MiB?
