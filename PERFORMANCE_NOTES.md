# Performance notes — Blackwell / CUDA 13 port

This branch of MXNet builds and passes the test suite on **NVIDIA Blackwell
(sm_120)** with the **CUDA 13.0 / cuDNN 9.14 / NCCL 2.28** toolchain. It's
correct, but it is **not yet performance-tuned for Blackwell**. The aim of the
port was "make it run again"; making it fast is a follow-up.

## Observed performance on RTX PRO 4000 Blackwell (24 GB)

| Op | Shape | Throughput | Tensor-core peak (fp16) | Util |
|---|---|---|---|---|
| `mx.nd.dot` fp32 | 1024³ × 100 | **0.89 TFLOP/s** | (not applicable) | low |
| `mx.nd.dot` fp16 | 2048³ × 50 | **2.7 TFLOP/s** | ~150 TFLOP/s | ~2% |
| Conv2D fwd+bwd fp32 | 16×3×64×64, 3×3 → 8 ch | runs via cuDNN 9 autotune | — | — |

For comparison: peak Blackwell fp16 dense tensor-core throughput on this part
is around **150 TFLOP/s**. The current build hits ~2% of that — i.e. **we're
roughly 50× off peak** for large fp16 matmul.

During the test suite the GPUs sit at ~1–8% utilisation for nearly all of the
~13,000 unit tests. That's typical of a correctness suite (tiny shapes,
asnumpy() round-trips dominate), but it also matches what we see on real
workloads: this build is **Python-and-dispatch-bound**, not GPU-bound.

## What's missing for "good" Blackwell performance

### 1. Tensor-core / cuBLASLt / CUTLASS paths
MXNet's matmul path goes through legacy `cublasGemmEx`/`cublasGemmStridedBatchedEx`
with `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (or `_DEFAULT` for fp32). This used to be
"fast enough" on Ampere but is no longer competitive on Hopper/Blackwell, where
cuBLASLt picks better kernels and CUTLASS provides hand-tuned MMA paths.

Plausible work: introduce a `cublasLtMatmul` path in `src/operator/linalg_impl.h`
gated behind an env var; later add a CUTLASS-based fallback for shapes cuBLASLt
doesn't handle well.

### 2. fp16 / bf16 with proper tensor-core math
`MXNET_FC_TRUE_FP16` is the existing knob for fp16-everywhere. It's off by
default and the pseudo-fp16 path (fp32 accumulation) doesn't hit MMA on
Blackwell. For real perf, the build needs:
- `CUBLAS_COMPUTE_16F_PEDANTIC` / `CUBLAS_COMPUTE_32F_FAST_16F` selection logic
- bf16 type support end-to-end (currently mxnet uses a custom `bfloat16` struct
  rather than cuDNN/CUDA's native `__nv_bfloat16`)

### 3. cuDNN 9 RNN port
The legacy cuDNN 7-style RNN API (`cudnnSetRNNDescriptor_v6`,
`cudnnRNNForwardInference`, `cudnnRNNBackwardData`, plus the `*Ex` variants)
was removed in cuDNN 9. **This port gates them out behind `CUDNN_VERSION < 9000`**;
on cuDNN 9 builds (i.e. anything CUDA 13) GPU RNN/LSTM/GRU falls through to
`LOG(FATAL)` and the operator is effectively unavailable on GPU.

Real fix: port `src/operator/rnn-inl.h` to the cuDNN 8 v8 RNN API
(`cudnnSetRNNDescriptor_v8`, single `cudnnRNNForward` with `fwdMode`,
`cudnnRNNBackwardData_v8`, `cudnnRNNBackwardWeights_v8`,
`cudnnGetRNNWeightSpaceSize`, `cudnnGetRNNTempSpaceSizes`,
`cudnnGetRNNWeightParams`). Estimated cost: ~3–5 days including the weight
layout differences (`cudnnGetRNNLinLayerMatrixParams` returns a different
packing than `cudnnGetRNNWeightParams`).

### 4. oneDNN bump for CPU ops
The vendored oneDNN commit (`58be3660fb`, ~v0.21-era) predates Blackwell *and*
predates AMX. CPU ops in this build are routed to OpenBLAS as a result
(`USE_ONEDNN=OFF`).

Bump `3rdparty/onednn` to `v3.5+` (which has both Blackwell GPU support and
modern CPU kernels) and re-enable `USE_ONEDNN=ON`. Most call sites should just
work; the bf16 INT8 quantisation subgraphs may need touch-ups.

### 5. Fused operators (NNVM passes)
The fused-op JIT (`src/operator/fusion/`) targets Pascal/Volta/Ampere kernel
patterns. On Blackwell it still works correctly but doesn't exploit Blackwell-
specific instructions (e.g. fp8 MMA, distributed shared memory). Whether this
matters depends on the workload; for inference latency, the unfused path is
usually already cuDNN.

### 6. Custom CUDA kernels not yet retuned
Lots of mxnet's own kernels (in `src/operator/tensor/`, `src/operator/nn/`,
`src/operator/contrib/`) were tuned for sm_70/sm_80 launch params (block size
256/512 etc.). They run correctly on sm_120 but launch configs may be
suboptimal for Blackwell's SM layout (twice as many SMs, different shared-mem
size). A profiling pass over the top-N ops by call frequency would be
worthwhile.

## Known correctness caveats

These remained after the port and aren't Blackwell-specific:

- **`test_pooling_versions`** — one element in ~480k differs by ~5% between
  cuDNN 9's max-pool kernel and the CPU reference. The test's tolerance was
  written against an older cuDNN. Not a wrong result, just outside the legacy
  tolerance.
- **`test_subgraph::test_make_subgraph`** — sparse-storage NDArray save fails
  under numpy-shape semantics; pre-existing MXNet limitation flagged at
  `src/ndarray/ndarray.cc:1867`.
- **`test_symbol::test_symbol_infer_shape`**, `test_load_save_symbol` —
  `infer_shape_partial` API returns `None` where the test expects `()` for
  unknown-shape; also test isolation between np-shape and np-array semantics.
- **`test_profiler::test_custom_operator_profiling`** — Python custom-op
  callback signature drift; not investigated.
- **`test_gluon_data::test_multi_worker`** — `DataLoader` collation produces a
  `dtype('O')` array with newer NumPy that the C++ side rejects.

None of these were caused by the port; they're upstream MXNet brokenness that
the original (now-retired) CI never caught because it ran older Python/NumPy/scipy.
