# Changelog

All notable changes to this fork of Apache MXNet are documented in this
file. This fork lives at [`smolix/mxnet`](https://github.com/smolix/mxnet)
and exists to port MXNet 2.0 to Blackwell (sm\_120) / CUDA 13 / cuDNN 9 /
oneDNN v3, since upstream Apache MXNet was archived on 2023-11-17.

**Upstream base**: `apache/mxnet` `release-2.0` at commit `dd7553781`
(the first commit in this fork). Upstream archived 2023-11-17 with no
Blackwell or CUDA 13 support; this fork is the only place that work will land.

Version string format: `<upstream>+cu<cuda-major>.bw.<YYYYMMDD>`.

---

## Current state

The latest published wheels are `v2.0.0+cu13.bw.20260614` (Linux/CUDA, `cp311` +
`cp312`) and `2.0.0+cpu.macos.20260614` (Apple Silicon). For a thematic summary of
everything this fork changed see [`FIXED.md`](FIXED.md); for known limitations and
open work see [`OPEN_ISSUES.md`](OPEN_ISSUES.md). The dated sections below are the
detailed per-release record through `20260518`; later releases
(`20260529` … `20260614`) are summarized in `FIXED.md` and on the
[Releases page](https://github.com/smolix/mxnet/releases). The bullets in this
section describe notable `master` changes after `20260518.2`.

### Port and dependency coverage

- **CUDA 13** — the active fork target remains CUDA 13 with Blackwell `sm_120`
  support and a broader release-build arch policy of sm\_80 / sm\_86 / sm\_89 /
  sm\_90 / sm\_120 when artifact size allows. Linux validation is currently on
  Ada `sm_89` hardware with CUDA 13.0; Blackwell performance claims still need
  a dedicated `sm_120` rerun before a new public release.
- **cuDNN 9** — the port uses cuDNN 9 APIs, including the v8 RNN path and TF32
  by default for FP32 convolution. cuDNN 9.22 remains the recommended runtime
  because earlier 9.x builds have weaker `sm_120` heuristic coverage; frontend
  autotune is still env-gated.
- **oneDNN v3** — oneDNN v3.11 is the CPU backend baseline. The release notes
  now call out the v3 INT8 scale-direction change, batch-norm coverage, BF16
  fallback on hosts without AVX-512-BF16, and AArch64 fallback guards for
  oneDNN paths that are not reliable on Apple Silicon.
- **Quantization and QAT** — post-training INT8 inference paths remain the
  supported path. `quantize_v2` has an STE backward and
  `quantize_net(..., qat=True)` preserves trainable parameter gradients, but
  full QAT through oneDNN subgraph ops is not implemented: `_sg_onednn_conv`
  and `_sg_onednn_fully_connected` still need real backward bodies before QAT
  training-through-quantized-graph can be advertised.

### Validation and release caveats

- **Apple Silicon** — native macOS arm64 CPU-only builds are validated as a
  separate, non-CUDA profile. The optimized C++ suite passed 89/89, a broad
  Python sweep reached 14,045 pass / 67 skip with one known DataLoader failure
  before the final fallback, targeted post-fix DataLoader checks passed, and a
  slim `2.0.0+macos.arm64.20260520` wheel smoke imported successfully.
- **Linux x86 / CUDA** — the current CUDA 13 validation build imports with
  CUDA, cuDNN, NCCL, oneDNN, and four RTX 4090 GPUs visible. Focused CUDA,
  oneDNN, packaging, lifecycle, quantization, and GPU regression checks have
  passed locally, but the broad GPU/operator rerun is still incomplete.
- **Packaging provenance** — the public `20260518.2` wheel predates later
  source fixes and must not be cited as current-source provenance. New release
  artifacts should be rebuilt from a clean tree with an explicit
  `MXNET_PACKAGE_VERSION`, recorded commit/submodule/build options, OpenCV-off
  CUDA wheel policy unless all OpenCV shared libraries are bundled, and a clear
  distinction between raw `linux_x86_64` wheels and any future manylinux claim.

### Code-review hardening — user-visible changes

A from-scratch code review (summarized in [`FIXED.md`](FIXED.md)) drove a batch of correctness,
safety, and robustness fixes. Most are internal (capture-safety guards, 64-bit
index discipline, exception safety, RAII, lock-free engine paths, build/tooling).
The user-facing ones:

- **`NDArray` is now unhashable.** Legacy `mx.nd.NDArray` sets `__hash__ = None`
  (matching the `mx.np` frontend), because equality is element-wise. Using an
  `NDArray` as a dict key or set member now raises `TypeError` instead of hashing
  by object identity. Use `id(arr)` explicitly if you need identity keying.
- **NumPy integer scalars accepted as indices/integer args.** `integer_types`
  now includes `numpy.integer`, so `np.int8`/`np.int16`/`np.uint*` scalars work
  where they previously raised. Strictly widens accepted input.
- **BatchNorm running statistics update in the training-mode forward pass**
  (PyTorch-style). A forward-only training pass now advances `running_mean` /
  `running_var`; previously the update happened during backward.
- **Out-of-bounds index semantics.** `gather_nd` backward now *drops* OOB indices
  instead of silently wrapping them (modulo); `index_add` / `index_update` skip
  OOB indices in-kernel and no longer pay a per-call host validation+sync by
  default. Set `MXNET_INDEX_BOUNDS_CHECK=1` to restore the eager host-side
  bounds check (raises on OOB).
- **Quantized `elemwise_mul` int32 output rounds to nearest** (`nearbyint`)
  instead of truncating toward zero; float output is unchanged.
- **`histogram` fp16 bin placement near an edge** now matches NumPy/CPU.
- **npy/npz loading rejects malformed/crafted archives** with graceful errors
  instead of undefined behavior (hardens `.npz` / `.params` loading of untrusted
  input).
- **Cleaner error propagation.** Array-coercion paths no longer swallow
  `KeyboardInterrupt` / `SystemExit` and now chain the original cause
  (`raise ... from e`).
- **Engine config.** Pushing an op with duplicate dependency vars now raises a
  clear fatal error in release builds (previously a nondeterministic hang). The
  `MXNET_ENGINE_TYPE` `Async` tag is honored only as a suffix, matching engine
  construction.

### Performance

- **Cython fast path restored.** The `_cy3` Cython extension never compiled —
  `cython/ndarray.pyx` used the Python-2 name `long`, so `cythonize` errored and
  every build (including shipped wheels) silently fell back to the slow ctypes
  marshaling path for *every* imperative op. Fixed the compile, and gave the
  cython `NDArrayBase` a `__del__` finalizer (PEP 442) matching the ctypes class
  so NDArrays in reference cycles free their backend handle on cyclic-GC
  collection (the missing finalizer otherwise leaked handles). The wheel now
  builds with `MXNET_WITH_CYTHON=1`, dropping per-op dispatch to the Cython fast
  path — a measurable speedup for op-heavy / small-op eager workloads.

---

## 2.0.0+cu13.bw.20260518 — 2026-05-18

Second Blackwell release. Builds on `20260517`; adds upstream bug-fixes, a
cuBLASLt path (PR-A fp32 + PR-B fp16/bf16/fp64), cuDNN frontend autotune,
INT8 concat routing via `TmpMemMgr`, oneDNN v3 BF16 fallback, and a full
post-release correctness + performance sweep. 99.8% pass rate on the
validated test surfaces (1324 pass / 3 fail / 296 skip).

### Resolved upstream bugs

| apache# | Summary | Commit |
|---------|---------|--------|
| [#21199](https://github.com/apache/mxnet/issues/21199) | 5D-input 1×1 conv reorder `pd-creation failure` on oneDNN v3 | `c8ccd53e8` |
| [#19353](https://github.com/apache/mxnet/issues/19353) | `linalg_impl.h` temp-buffer freed before GPU kernels finish → NaN | `251980078` |
| [#19019](https://github.com/apache/mxnet/issues/19019) | AMP weight cast not cached on recursive networks → GPU OOM | `f5e7c063c` |
| [#18865](https://github.com/apache/mxnet/issues/18865) | `mx.random.seed` order-dependent on multi-CPU context | `c2df8dd44` |
| [#18751](https://github.com/apache/mxnet/issues/18751) | BatchNorm `running_mean`/`running_var` swapped on GPU in forward | `a47ce39d9` |
| [#18584](https://github.com/apache/mxnet/issues/18584) | `batch_dot` fp16 precision diverges from `dot` on GPU | `08cb44d1d` |
| [#18564](https://github.com/apache/mxnet/issues/18564) | GPU memory profiler: tensordot attr name mismatch under oneDNN v3 | `ed6757e64` |
| [#18090](https://github.com/apache/mxnet/issues/18090) | Engine teardown deadlock / CI hangs after last test | `292246c06` |
| [#17495](https://github.com/apache/mxnet/issues/17495) | `Profiler::Get()` DCLP data race on non-atomic `shared_ptr` | `085da5e09` |
| [#16686](https://github.com/apache/mxnet/issues/16686) | `grad_req='add'` numerical inconsistency vs manual accumulation | verified fixed (no C++ change needed) |
| [#14264](https://github.com/apache/mxnet/issues/14264) | `nd.reshape` to smaller shape silently truncates instead of raising | `7d958459c` |
| [#13915](https://github.com/apache/mxnet/issues/13915) | `test_activation` softrelu backward flake | `8f6cc19ad` + cuDNN 9.22 bump |
| [#11163](https://github.com/apache/mxnet/issues/11163) | Engine destructor `notify_all` races with interpreter teardown | `292246c06` |
| [#20447](https://github.com/apache/mxnet/issues/20447) | In-place ops (`+=`, `-=`, …) silently change `lhs` dtype | `c86d09306` |

### New Blackwell port features

- **CUDA 13.0 / cuDNN 9.22 / NCCL 2.28.3** — full Blackwell sm\_120 build
  with multi-arch SASS (sm\_80, sm\_86, sm\_89, sm\_90, sm\_120 + PTX 120
  fallback). cuDNN bumped 9.14 → 9.22 for sm\_120 heuristic coverage;
  depthwise 3×3 256→256 went 0.16 → 1.14 TFLOPS (~7×).
- **TF32 default ON** for FP32 conv on cuDNN 9 (sm\_120): measured **2.87×**
  speedup (3×3 28×28 256→256 batch 32: 14.46 → 41.48 TFLOPS). Mirrors
  PyTorch / TF defaults. (`783cfa133`)
- **oneDNN v3.11 port** — complete mechanical API drift sweep across every
  primitive in `src/operator/nn/dnnl/`. Full INT8 path: per-OC weight scales
  on `DNNL_ARG_WEIGHTS`, fused conv/FC/sum, dequant-to-fp32 output. Batch
  norm fwd+bwd complete. (`273d03ec5`, `960be41c0`, and ~15 follow-up commits)
- **oneDNN v3 BF16 fallback** — `DNNLISASupportsLowpFloat()` detects
  non-AVX-512-BF16 hosts at runtime and upcasts bf16 operands to fp32, then
  reorders back. Unblocks the 6 `test_amp_subgraph.py` inner\_product
  creation failures on Zen2/Skylake/IceLake CPUs. (`a78d29355`)
- **cuBLASLt PR-A + PR-B** — `MaybeCublasLtSgemm` (fp32), `Hgemm` (fp16),
  `Bf16Gemm`, `Dgemm` wrappers in `src/operator/linalg_impl.h`. Per-device
  LRU heuristic cache (cap 256), 32 MiB workspace. Env-gated
  (`MXNET_USE_CUBLASLT=1`), default off pending numerics audit.
  (`75232ca9b` PR-A, `05af4d576` PR-B)
- **cuDNN frontend autotune** — `UseFrontendAutotune()` unions
  `CUDNN_HEUR_MODE_A + MODE_B` candidate plans (20–23 vs fewer from a single
  mode). Env-gated (`MXNET_CUDNN_AUTOTUNE_FRONTEND`). (`e0eb106ea`)
- **AMP weight-cast cache** — `_cast_symbol_NDArray` caches the fp16 result
  keyed by `(id(src), src_dtype, dst_dtype)`; shared-layer loops reuse one
  buffer instead of allocating a new one each step. Public
  `clear_weight_cache()` API. (`f5e7c063c`)
- **Engine clean shutdown** — `MXNotifyShutdown()` now calls
  `Engine::Get()->Stop()` (joins all worker threads) before interpreter
  teardown; eliminates CI hangs and the Linux `dlclose` race. (`292246c06`)
- **cuDNN v8 RNN API** — `cudnnRNNForward`, `cudnnRNNBackwardData_v8`,
  `cudnnRNNBackwardWeights_v8`, `cudnnSetRNNDescriptor_v8` replace the removed
  v7 training/inference functions. (`817f5bea1`)
- **QAT STE backward** — `quantize_v2` gets a Straight-Through Estimator
  backward; `quantize_net(..., qat=True)` keeps `grad_req='write'` on all
  params. (`ba635eb2d`)
- **PyTorch-style pip-deps wheel** — no bundled CUDA/cuDNN/NCCL libs under
  `mxnet/lib/`; declares `nvidia-cudnn-cu13>=9.22` and `nvidia-nccl-cu13>=2.28`
  as `install_requires`. **Wheel size: 2.22 GB → 454 MB (79% reduction)**.
  RUNPATH points to system CUDA 13 install. (`83718e389`)
- **21 upstream-disabled tests re-enabled** — audited GREEN on Blackwell.
  (`cedeb2f9b`)
- **`MXlib.__del__` libdl.so fallback** — handles glibc 2.34+ which absorbed
  libdl into libc. (`c57970216`)

### Known open issues

- **#4** — `test_pos_single_concat_pos_neg[int8/auto-data_shape1]`: entire
  output channels zeroed; suspect oneDNN v3 uint8→int8 reorder in
  `dnnl_quantized_concat.cc`. Needs `DNNL_VERBOSE=2` trace.
- **#5** — `_sg_onednn_fully_connected` / `_sg_onednn_conv` still have
  `FGradient = MakeZeroGradNodes`; QAT STE is in place but won't propagate
  through fused subgraph ops until proper backward support lands.
- **#7** — BF16 silently falls back to fp32 on Zen2/Skylake (no AVX-512-BF16).
  Hardware-bound; not fixable in software.
- **#14** — ONNX export/import errors at collect time; module path never
  updated for MXNet 2.0 numpy ops.
- **#19** — cuBLASLt PR-C (stride-aware), PR-D (INT8), PR-E (default-on)
  deferred to next revision; env-gated PR-A/B correct.
- **#23** — FP32 deconvolution hard-codes `CUDNN_DEFAULT_MATH`; TF32 not
  enabled for transposed-conv-heavy nets (GAN generators, super-resolution).
  Patch ready at `.investigations/n23_tf32_patch.patch`.
- **#24** — Small/bandwidth-bound fp16 ops (softmax, LayerNorm <4K, elementwise
  add/mul <1M) are 2–6× slower than PyTorch; root cause is pre-existing
  MXNet multi-pass kernels + NDArray dispatch overhead, not a Blackwell
  regression.
- **#26** — Distributed training (KVStore PS, Horovod) not exercised beyond
  2-GPU NCCL smoke test.
- **#49** — `test_self_attention[split=True-*]` (12 parametrizations in
  `test_matmul_subgraph.py`): int8 quantization failures + one segfault
  (RC=139). Likely common root cause with #4.

---

## Breaking changes vs upstream apache/mxnet 2.0

| Change | Detail |
|--------|--------|
| **CUDA 13 required** | `libmxnet.so` linked against CUDA 13.0; will not load on CUDA ≤12 drivers. |
| **sm\_120 primary target** | Multi-arch fatbin includes sm\_80/86/89/90/120; PTX 120 fallback for unknown future SM. |
| **Python 3.11 / 3.12 wheels** | Released wheels are `cp311` and `cp312` (`linux_x86_64`); 3.13 untested. |
| **numpy < 2 recommended** | `setup.py` pins `numpy>=1.17`; numpy 2.x API drift causes test failures. Pin `numpy<2` until audited. |
| **RUNPATH-based CUDA loader** | No bundled CUDA runtime under `mxnet/lib/`; requires system CUDA 13 at `/usr/local/cuda/` and pip-installed `nvidia-cudnn-cu13>=9.22`, `nvidia-nccl-cu13>=2.28`. |
| **oneDNN v3 INT8 scale direction** | Per-OC scales now bound on `DNNL_ARG_WEIGHTS` (not DST); DST scale divides in v3 vs multiplies in v2. Custom quantized operators referencing internal scale args must be audited. |
| **cuDNN v8 RNN API** | Legacy `cudnnRNNForwardTraining` / `cudnnRNNForwardInference` removed. Any code reaching into the cuDNN RNN descriptor path must use v8 functions. |

---

## Test coverage delta vs upstream

New regression tests added in this fork (cumulative across both releases):

| Test file | Tests | Guards |
|-----------|------:|--------|
| `test_batchnorm_running_stats.py` | 6 | apache#18751 |
| `test_batch_dot_fp16_parity.py` | 6 | apache#18584 |
| `test_linalg_temp_sync.py` | 5×200 | apache#19353 |
| `test_amp_weight_cache.py` | 7 | apache#19019 |
| `test_random_seed_order.py` | 6 | apache#18865 |
| `test_engine_shutdown.py` | 12 | apache#11163/#18090 |
| `test_inplace_dtype.py` | 4 | apache#20447 |
| `test_grad_req_add_consistency.py` | 6 | apache#16686 |
| `test_threaded_init.py` | 2 | apache#17495 |
| `test_embedding_backward_nan.py` | 4 | apache#11314 |
| `test_cublaslt_fc.py` | 12 | cuBLASLt dtype parity |
| `test_nccl_singleproc.py` | 10 | NCCL 2-GPU push/pull |
| `test_a1_5d_conv_reorder.py` | 5 | apache#21199 |
| Unskipped upstream tests | 21 | Audited GREEN on Blackwell |

**~106 new test cases** added in this fork. 21 previously disabled upstream
tests re-enabled after confirming they pass on Blackwell.

---

## 2.0.0+cu13.bw.20260517 — 2026-05-17

First Blackwell preview release. Forward inference for fp32 / fp16 / INT8 is
solid; quantized backward and several auxiliary paths remain open (see
[`OPEN_ISSUES.md`](OPEN_ISSUES.md)).

### Added

- Blackwell (sm\_120) support: CUDA 13.0, cuDNN 9.22.0, NCCL 2.28.3.
- oneDNN v3.11 with the full INT8 path enabled: per-OC weight scales, fused
  conv / FC, fused sum, dequant-to-fp32 output, fused activations.
- GitHub Release wheel for `linux_x86_64` (Python 3.11).
- `BUILDING.md` documenting the build recipe for the release wheel.
- An open-items list (correctness, functionality, performance, test coverage,
  build/release, code quality) — since consolidated into
  [`OPEN_ISSUES.md`](OPEN_ISSUES.md).

### Changed

- cuDNN RNN code rewritten against the cuDNN v8 API.
- INT8 plumbing: per-output-channel scales now bound on `DNNL_ARG_WEIGHTS`.
- Reorder paths route through `DNNL_ARG_SRC`; oneDNN v3 DST scale divides
  (v2 multiplied). `sum_scale` rescaled by `DST_scale` for fused conv/FC sum.
- FC f32-output dequantization now uses `DNNL_ARG_SRC` + `DNNL_ARG_WEIGHTS`
  scales with f32 bias (oneDNN v3 rejects s32 bias + f32 dst combination).
- Version string: `2.0.0+cu13.bw.20260517`.
- oneDNN submodule bumped from v2.x to v3.11.

### Fixed (selected)

- `dd7553781` Port to CUDA 13 / cuDNN 9 / sm\_120 — root build/link/SASS bringup.
- `273d03ec5` oneDNN v3: bump submodule + mechanical API drift sweep.
- `960be41c0` oneDNN v3: complete batch\_normalization fwd + bwd.
- `783cfa133` cudnn: enable TF32 by default for FP32 conv on cuDNN 9 (2.87× on sm\_120).
- `f103c5491` deps: bump cuDNN 9.14 → 9.22 (depthwise 3×3 ~7× faster on sm\_120).
- `f5934f094` quantize\_graph\_pass: fall back to None when calib range is NaN.
- `83718e389` release: switch to PyTorch-style pip-deps wheel (2.22 GB → 454 MB).
- `1d2198862` adaptive\_avg\_pool: force CPU-reference fallback for backward (72/72 PASS).

### Known issues at 20260517 (most resolved in 20260518)

See [`OPEN_ISSUES.md`](OPEN_ISSUES.md) for the full list. Notable items carried forward:
int8 concat numerical error (#4), quantized subgraph backward (#5), ONNX
export/import broken (#14), deconvolution TF32 gap (#23).

---

For Apache MXNet's pre-archive history see [`NEWS.md`](NEWS.md).
