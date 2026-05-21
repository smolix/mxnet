# MXNet Port Issues

Updated: 2026-05-21
Current branch: `master`
Current head: local validation commits ahead of `origin/master` (see git log)
Apple Silicon follow-up merge: PR #28 from `followup/full-sweep-macos-wheel`
Linux validation host: 4x RTX 4090 Ada (`sm_89`), CUDA 13.0, cuDNN/NCCL
host dependencies and submodules installed; CUDA `sm_89` build configured; first
`mxnet` build exposed an avoidable `operator_tune.cc` compile in non-tuning
builds. Validation builds with `USE_OPERATOR_TUNING=OFF` now exclude that source
and link `build/libmxnet.so`;
editable `.venv-mxnet` imports the local CUDA/oneDNN/NCCL/cuDNN library and
reports all 4 GPUs. A oneDNN `batch_dot` descriptor bug found by the d2l
transformer repro is fixed locally and covered by a focused regression test.
Broad CPU/GPU sweeps were restarted under tmux on 2026-05-21. The CPU unittest
lane completed once and its failures were reduced to optional extension artifacts
and GPU-memory pressure in tests that pass in isolation; a later lower-concurrency
CPU rerun was stopped after the rebuilt binary changed. The full GPU lane aborted
in the fork-safety DataLoader test before producing a complete summary; that
focused crash is fixed, and the first failed GPU files pass or skip cleanly in
isolation. The DNNL adaptive-pooling numeric-gradient matrix has been reduced
after the row-sparse timeout, and the focused adaptive-pooling check now passes.
Local DNNL quantized conv+sum mitigation work is present and rebuilt, and its
focused regression now passes. A host reboot realigned the NVIDIA 580
server-open kernel/userspace stack at `580.159.03`; direct GPU pytest commands
again reach CUDA devices, and focused GPU regressions for reducer, TF32 deconv,
fusion, deferred compute, NumPy einsum, histogram, and Proposal/MultiProposal
checked-arithmetic paths pass. The stale cuDNN stream regression now uses a
deterministic NumPy oracle and passes locally. The broad CPU/DNNL/GPU reruns
are still pending. Environment-prefixed commands can still run in a restricted
tool context without `/dev/nvidia*`, so CUDA error 304 from those probes should
be treated as a command-environment artifact unless it reproduces from the
direct pytest/Python invocation shape.
Local release tag: `macos-arm64-slim-wheel-20260520`

This file is a status index, not a changelog. Historical details live in git
commits and the retained investigation notes. Items are grouped by what a
maintainer needs to decide next.

Status labels:

- **Open**: known issue; no verified fix on the current branch.
- **In progress**: local work exists but is not committed and verified yet.
- **Deferred**: cannot be verified on the current machine or is deliberately out
  of scope for the current validation pass.
- **External**: owned by D2L, notebook infrastructure, or another project unless
  a current MXNet runtime failure is reproduced.
- **Informational**: retained context; not a work item.
- **Resolved**: fix is committed or otherwise verified; retained here only when
  it is useful context for future work.

---

## Immediate Linux/CUDA Execution Queue

This is the ordered queue for the current Linux/Ada host. Blackwell-specific
performance claims still need a later dedicated Blackwell run, but Linux/CUDA
correctness and cuDNN/CUDA 13 behavior can be validated here.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| L0 | Resolved | Build setup | Host toolchain/runtime packages are installed, submodules are initialized, and CUDA 13 `sm_89` validation builds with `USE_OPERATOR_TUNING=OFF` link `build/libmxnet.so` without compiling `operator_tune.cc`. `.venv-mxnet` imports the local library and reports CUDA, cuDNN, NCCL, oneDNN, and 4 visible GPUs. | Use this validation build for focused Linux/Ada tests; keep intentionally enabled operator tuning as a separate release-build policy choice. |
| L1 | In progress | Apple fixes on x86 | Apple Silicon fixes for lifecycle, DataLoader, DLPack, quantization fallbacks, oneDNN test harnesses, and NumPy drift need validation on Linux x86 with oneDNN enabled. DataLoader fork-safety, BF16 skip policy, DNNL batch-dot coverage, OpenMP fork handling, extension optional-artifact behavior, DNNL adaptive-pooling triage, broad CPU unittest at `1752 passed, 91 skipped`, focused C++ lifecycle/DNNL/BatchNorm/operator filters, several focused GPU harness checks, DNNL core/subgraph/quantization serial reruns, and the DNNL quantized conv+sum regression now pass locally. | Shard the remaining broad GPU/operator lanes before calling Linux x86 validation complete. |
| L2 | Resolved | CUDA smoke | The d2l diagnostics were produced from a pre-Apple-port wheel that lacked Ada kernels. After rebuilding for `sm_89`, the standalone d2l GPU probes are OK across the 4-GPU host. | Treat the old no-kernel-image failures as stale for this host; use current D2L notebook-run/output-audit artifacts under L4 for any notebook follow-up. |
| L3 | In progress | CUDA regression batch | Targeted CUDA tests have cleared cuDNN/TF32 deconv, cuDNN large-channel conv/deconv probes, cuDNN frontend-autotune smoke and explicit no-plan fallback, cuDNN stream/workspace regression coverage, cuBLASLt GEMM/FC/dtype/strided checks plus a same-process 4-GPU threaded cuBLASLt stress, fp16 batch-dot parity, linalg temp storage, deferred-compute GPU, reducer regressions, NumPy einsum GPU, KVStore local/device/GPU checks, NCCL multiprocess, histogram GPU edge cases, Proposal/MultiProposal checked-arithmetic coverage, a 26-node bounded GPU operator micro-shard, and the NCCL single-process bandwidth check after converting the hard bandwidth gate into a metric. Extension GPU tests now skip when optional shared libraries were not built. | Shard the full GPU lane and run the remaining focused CUDA backlog under C4. Use direct pytest/Python commands for GPU access in this tooling context. |
| L4 | External | D2L artifact intake | The rebuilt `sm_89` runtime clears the standalone GPU probe gate, and the transformer standalone repro now passes after the oneDNN `batch_dot` descriptor fix. The old d2l notebook failures came from stale artifacts and should not reopen MXNet work by themselves. | Consume current D2L notebook-run/output-audit artifacts when available; reopen MXNet work only for concrete fresh runtime repros, not stale stamps or dead-kernel summaries. |
| L5 | Resolved | Tracker cleanup | Duplicate issue trackers and stale markdown reports have been imported here or removed from the repo. Current D2L diagnostic logs are treated as local artifacts; only executable repro tools should be retained intentionally. `origin/docs/handover-2026-05-19` is an ancestor of `master` with no branch-only diff; the useful handover facts are already retained under B4, O4, and O11. | Keep `issues.md` as the processing queue; retain only active investigation notes and executable d2l repro tools. |
| L6 | In progress | Compiler noise | GCC 13/NVCC CUDA 13 builds emit enough warning noise to hide real failures. The dmlc optional cluster, product/nanprod reduction cleanup, local MXNet tuple stack-initialization, runtime-handle cleanup, CN4 min/max residual initialization, CN5 unsigned CUDA guards, CN6 local sentinel conversions, CN7 half parameter packing, mshadow packet allocation, einsum initialization, CTC include-boundary, half max-pool initialization, cheap local cleanup noise, and avoidable non-tuning `operator_tune.cc` compilation are now reduced locally. | Continue with remaining high-volume or policy clusters: GCC 13 tuple/ADT array-bounds false positives, bundled dmlc queue offsets, and oneDNN ITT executable-stack note. |
| L7 | In progress | Test scheduling | Parallel full-suite lanes can overload the host if CPU xdist, C++ gtest, oneDNN numeric-gradient tests, and GPU operator sweeps overlap. The target load envelope for this machine is about 48-64 runnable tasks. | Keep one heavy CPU lane active at a time, add GPU shards when memory is idle, and pause/resume long DNNL work instead of killing it when load spikes. |
| L8 | In progress | Build freshness | Local DNNL quantized conv+sum and compiler-noise edits rebuilt successfully; `build/libmxnet.so` was rebuilt again after the host reboot and submodule cleanup at 2026-05-21 14:47:01 UTC. The library was incrementally rebuilt again after runtime-container, CUDA Proposal/MultiProposal hardening, and the non-tuning `operator_tune.cc` build fix. Focused post-rebuild checks for reducer, packaging, optimizer, CTC, pooling, einsum, TF32 deconv, cuDNN frontend-autotune smoke, cuDNN stream/workspace behavior, cuBLASLt serial and threaded stress, GPU reducer paths, CUDA smoke/regression subsets, DNNL serial shards, broad CPU unittest, runtime containers, Proposal/MultiProposal, embedding default coverage, and focused C++/gtest filters pass. The full GPU/operator lane is still incomplete, and CMake still embeds the older `MXNET_COMMIT_HASH` value from `b881a9c5a` because the build was not reconfigured. | Finish broad GPU reruns on this rebuilt library; reconfigure before packaging, publishing artifacts, or answering binary provenance questions so embedded metadata and wheel tags match the release commit. |
| L9 | Resolved | Host GPU driver state | The CUDA error 304 / invalid-context failures came from a host NVIDIA mismatch after `/usr/bin/unattended-upgrade` upgraded the NVIDIA 580 server-open userspace packages from `580.126.20` to `580.159.03` while the loaded kernel module stayed at `580.126.20`. After reboot, `nvidia-smi` reports driver/userspace `580.159.03`, all four RTX 4090s are visible, and direct GPU pytest invocations pass focused smoke/regression checks. CUDA error 304 can still be produced by environment-prefixed probes that run without `/dev/nvidia*` inside the restricted tool context; that is not the host driver state. | Treat L9 as closed for this rebooted host. Use direct pytest/Python commands for GPU validation, and only reopen if the same CUDA failure reproduces in that command shape. |

---

## Compiler Noise Triage

These clusters came from the first Linux CUDA 13 `sm_89` build attempts on GCC
13/NVCC 13. Validation builds with `USE_OPERATOR_TUNING=OFF` completed
and linked `build/libmxnet.so`; the warning stream is in `build/mxnet-build.log`.
The goal is to reduce warning volume without papering over real CUDA or numeric
bugs.

| ID | Status | Area | Cluster | Triage action |
|---|---|---|---|---|
| CN1 | Resolved | Build throughput | Non-tuning builds no longer compile `src/operator/operator_tune.cc`: `tuned_op::UseOMP` is inlined to the existing non-tuning `true` behavior, CMake removes the source unless both `USE_OPERATOR_TUNING` and `USE_OPENMP` are enabled, a throwaway CPU-only non-tuning build linked, the main CUDA non-tuning build linked, and both build metadata scans omit `operator_tune.cc`. | Keep deliberately enabled operator-tuning release builds unchanged; reopen only if a release profile requires tuning and the remaining compile cost is unacceptable. |
| CN2 | In progress | Tuple/runtime allocation | Repeated GCC 13 array-bounds warnings flow through `Tuple`, `NDArray`, `TBlob`, `nnvm::NodeEntry`, and `std::shared_ptr` inline frames, but the root is MXNet runtime `InplaceArrayBase` / `SimpleObjAllocator` flexible-tail allocation for `ADTObj`. This is likely a compiler false positive against a deliberate layout pattern, with a separate hardening opportunity around `num_elems * sizeof(ElemType) + sizeof(ArrayType)` overflow. The local overflow guards rebuilt and passed `RuntimeContainer.*` plus `test_ffi_container.py`; they harden real arithmetic overflow but are not expected to eliminate every GCC 13 flexible-tail false positive. The parallel NNVM tuple cache initializer still belongs in a reachable TVM submodule commit or explicit third-party patch policy. | Decide whether release `RelWithDebInfo` builds need a narrow GCC 13 `-Wno-error=array-bounds` policy while keeping the warning visible. |
| CN3 | Resolved | dmlc optional | `include/dmlc/optional.h` emitted uninitialized warnings around `optional<T>().swap(*this)`, stream extraction, nullopt assignment, and scalar specializations. | Fixed in dmlc-core commit `d610d79` by using lifetime-aware assignment/reset/swap and by assigning parsed values only after successful extraction; standalone optional smoke passed. |
| CN4 | Resolved | Reductions | Half/bfloat broadcast-reduce residual initialization now initializes `val` and `residual` independently before reducer-specific setup, and boolean product/nanprod initializes unused residual storage and uses logical AND for bool reductions on CPU and RTC GPU paths. | Clean rebuild no longer shows the CN4 residual cluster; focused CPU reducer regressions passed. GPU/autograd-covered follow-up waits for L9. |
| CN5 | Resolved | CUDA type guards | Unsigned comparison warnings are reduced with type-trait negative checks in bincount, delete, nan-to-num, and NumPy random normal/location-scale validation paths. | Clean rebuild no longer shows the local CN5 unsigned-comparison cluster. Forward-only `bincount`, `delete`, `np_random`, and `np_randn` checks passed; autograd-covered random/nan-to-num checks wait for L9. |
| CN6 | Resolved | Sentinel conversions | Local MXNet `size_t` sentinel initialization warnings in `np_cross` and `np_matmul` are fixed by value-initializing vectors instead of filling them with `-1`. Bundled dmlc queue offset warnings are third-party boundary work, not local CN6 blockers. | Clean rebuild no longer shows local `np_cross`/`np_matmul` sentinel warnings. `test_np_matmul_error` passed; autograd-covered `np_cross`/`np_matmul` follow-up waits for L9. |
| CN7 | Resolved | Half param packing | Fused optimizer parameter packing in AdamW/AdaBelief-style code now uses typed assignment instead of `memcpy` into `mshadow::half::half_t` arrays, removing the local `-Wclass-memaccess` cluster. | Focused AdamW/AdaBelief tests passed; keep fused optimizer coverage in the CPU/GPU smoke set. |
| CN8 | Resolved | Local cleanup | Cheap and low-risk local warnings are reduced: CUDA resize/transformer unused variables are removed, KVStore NCCL hidden-overload/buffer noise is cleaned up, pointwise fusion initializes the crossing-subgraph output slot, CTC wraps only the vendored moderngpu include for deprecated declarations, mshadow packet allocation initializes the compiler-visible pointer path, einsum paths initialize selected costs and pointer arrays, half CPU max-pooling uses `NegInfValue`, and runtime-handle cleanup is validated. | Post-cleanup rebuild completed; warning scan no longer shows the mshadow packet, CTC/moderngpu, einsum, or half `numeric_limits::infinity()` pooling clusters. Focused CTC/einsum/pooling/reducer/TF32 tests passed. |
| CN9 | Open | Third-party/link boundaries | Bundled dmlc concurrent queue still assigns `-1` into a `uint32_t` sentinel under NVCC, and oneDNN's vendored ITT assembly still lacks a non-executable-stack note. Both local patches are upstreamable, but they would dirty detached submodules rather than produce reachable MXNet commits. | Do not commit private submodule-local fixes. Track as upstream/submodule policy work or carry only through an explicit third-party patch mechanism. |

### Compiler Noise TODO Clusters

- [x] CN2 tuple/ADT allocation: local source overflow hardening rebuilt and
      passed focused runtime-container/FFI checks.
- [ ] CN2 warning policy: still treat the GCC 13 `-Warray-bounds` cluster as
      likely flexible-tail allocation analysis noise and decide whether release
      builds should keep this as warning-only rather than `-Werror`.
- [x] CN4 reductions: product/nanprod and min/max residual initialization are
      reduced locally; clean rebuild confirms no remaining CN4 residual cluster.
- [x] CN5 unsigned CUDA guards: bincount, delete, nan-to-num, and NumPy random
      normal/location-scale checks are reduced with type-trait guards.
- [x] CN6 sentinel conversions: local `np_cross`/`np_matmul` sentinel
      initialization is fixed locally; leave bundled dmlc concurrent queue
      offsets to CN9 third-party boundary handling.
- [x] CN8 local cleanup: cheap unused-variable, KVStore NCCL, pointwise fusion,
      half parameter packing, mshadow packet allocation, einsum initialization,
      CTC include boundary, and half max-pool initialization are reduced locally;
      clean rebuild and focused tests pass.
- [x] CN1 build throughput: non-tuning builds inline the existing `UseOMP`
      behavior, exclude `operator_tune.cc`, and link in both throwaway CPU-only
      and main CUDA validation builds.
- [ ] CN9 third-party/link boundaries: treat bundled dmlc queue offset warnings
      and the oneDNN ITT executable-stack linker note as upstream/submodule
      policy items, not runtime correctness blockers.

---

## D2L Diagnostics Import

These items were imported from the prior d2l diagnostics reports and logs. They
were observed with MXNet `2.0.0+cu13.bw.20260517` before the Apple Silicon merge
and before an Ada-specific rebuild. Current triage keeps only D1 as an MXNet
packaging work item pending rebuilt-wheel artifact audit; D5/D6 are external
D2L notebook-run/output-audit ownership, D7 is external D2L/import-environment
work unless a standalone MXNet crash repro appears, and D8 is an informational
artifact-quality signal rather than an MXNet defect.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| D1 | In progress | Runtime deps | The prior MXNet wheel linked against system OpenCV 4.6 runtime libraries but did not include dependency metadata or bundled libraries, causing import-time `libopencv_imgcodecs.so.406` failures on clean hosts. Python wheel metadata cannot reliably express those system OpenCV SONAMEs. The primary packaging metadata/runtime-bundling guard is committed, the Linux release-wheel workflow now configures `-DUSE_OPENCV=OFF`, omits `libopencv-dev`, disables OpenCV dependency metadata with `MXNET_SETUP_ENABLE_OPENCV_DEPS=0`, and stages `libmxnet.so` at the package path that `libinfo.find_lib_path()` searches. The legacy `tools/pip/setup.py` CD path now also declares `opencv-python` metadata when OpenCV is enabled and bundles resolved `libopencv_*` libraries unless `MXNET_SETUP_ALLOW_SYSTEM_OPENCV=1` is set. Focused packaging tests passed `8 passed`. The current local `build/libmxnet.so` was configured `USE_OPENCV=OFF` and has no OpenCV DT_NEEDED entries, but there is no rebuilt wheel artifact under `dist/`, `python/dist/`, or `wheelhouse/` to audit. The current build metadata is stale and must be reconfigured before packaging; CUDA runtime bundling also needs `patchelf` available before a CUDA release-wheel audit. | Build the exact release wheel and artifact-audit it before closing D1. For the intended OpenCV-off release, verify extracted-wheel import succeeds, metadata does not require `opencv-python`, `readelf -d mxnet/libmxnet.so` shows no `libopencv_*`, and `mx.runtime.Features().is_enabled("OPENCV")` reports false. If shipping OpenCV enabled later, bundle every `libopencv_*` dependency or explicitly document required OS packages. |
| D2 | Resolved | CUDA arch coverage | The dominant d2l failure was `cudaErrorNoKernelImageForDevice` on RTX 4090 because the tested wheel did not include `sm_89` kernels. The local `sm_89` rebuild clears the standalone GPU probe gate on this Ada host. | Keep release-wheel architecture coverage open under O2/C4; no local Ada runtime action remains for the old wheel failure. |
| D3 | Resolved | GPU scalar host sync | Six d2l RNN/optimization notebooks reported `MXNetError: could not execute a primitive` when converting GPU scalar results to host/Python. The rebuilt standalone `gpu-scalar-to-host` and `gpu-gru-scalar-to-host` probes are OK. | If current D2L notebook-run/output-audit artifacts still show full-notebook failures, require a fresh MXNet runtime repro before reopening this standalone scalar path. |
| D4 | Resolved | Transformer native crash | The crash narrowed to oneDNN `batch_dot` using packed primitive descriptors to wrap MXNet default-layout buffers. Reordering inputs into the primitive-selected descriptors fixes the attention-shaped repro; `transformer-decoder-standalone` now passes with oneDNN enabled. The matching `_sg_onednn_batch_dot` path also needed a temp-space request for those reorders, and the existing subgraph batch-dot matrix now passes. | Keep `tests/python/dnnl/test_batch_dot_attention_regression.py` and `tests/python/dnnl/subgraphs/test_matmul_subgraph.py::test_batch_dot` in the oneDNN subset; reopen transformer work only if current D2L artifacts include a fresh MXNet runtime repro beyond the fixed standalone path. |
| D5 | External | Dead notebook kernels | `transformer.ipynb`, `natural-language-inference-bert.ipynb`, and `sentiment-analysis-rnn.ipynb` ended as `DeadKernelError` without useful Python traceback in the old diagnostics. The MXNet standalone transformer crash is fixed; rerunning those notebooks is assigned to the external D2L/notebook execution system. | Wait for current notebook-run/output-audit artifacts from that assigned system; reopen under MXNet only if a current runtime failure reproduces outside notebook infrastructure. |
| D6 | External | Notebook quality gate | `chapter_builders-guide/use-gpu.ipynb` had a passing stamp while stored outputs still contained MXNet GPU errors. This is an external D2L output-audit/notebook-runner issue unless current outputs show a fresh MXNet runtime failure. | Let the assigned notebook-run/output-audit system validate stamps against stored outputs; use any fresh MXNet errors it finds as concrete repro inputs. |
| D7 | External | D2L import-time GPU probing | In restricted environments, `d2l.mxnet` can query GPUs at import time through default arguments such as `devices=d2l.try_all_gpus()`. This remains external D2L/import-environment work unless a standalone MXNet crash repro appears. | Track as a d2l-side lazy-default fix; not an MXNet runtime bug unless MXNet itself crashes outside sandbox constraints. |
| D8 | Informational | Cross-framework artifact quality | Completed MXNet notebooks mostly had sane outputs; the main issue was missing GPU runtime coverage, not bad convergence in completed notebooks. Treat this as an artifact quality signal, not an MXNet defect. | Use current notebook-run coverage and output-audit artifacts as the quality signal after runtime fixes. |

---

## Current Full-Sweep Findings

These were observed on the 4x RTX 4090 Linux host on 2026-05-21. Treat
failures from broad concurrent lanes as triage inputs, not final verdicts,
because the host was under heavy CPU and GPU load. A lower-concurrency CPU
unittest rerun was stopped after the rebuilt binary changed, and no complete
post-fix GPU or DNNL core full-run summary is recorded here yet.

| ID | Status | Area | Finding | Next action |
|---|---|---|---|---|
| FS1 | Resolved | CPU unittest | The post-rebuild lower-concurrency broad CPU unittest lane passed with `1752 passed, 91 skipped, 65300 warnings` in `977.18s`, using `-n 4` and excluding the monolithic operator files. Earlier failures were either optional extension libraries now skipped cleanly or GPU OOM pressure that passed in isolation. | Keep this lower-concurrency CPU shard in the release validation matrix; track the excluded operator files through the dedicated GPU/operator lanes instead of reopening FS1. |
| FS2 | Resolved | oneDNN Python | Post-rebuild serial DNNL reruns now pass: `tests/python/dnnl/test_dnnl.py` passed `30`, DNNL subgraphs passed `935` with `16 skipped` and `4 xfailed`, `tests/python/dnnl/test_quantization_dnnl.py` passed `26`, and the generic quantization shard passed `26`. The earlier adaptive-pooling timeout was excess fallback work, and the reduced adaptive-pooling matrix plus DNNL conv+sum quantization checks pass. | Keep these DNNL subsets in the release validation matrix; broader INT8/self-attention and quantized-Gluon matrix work remains tracked under B1/B2. |
| FS3 | Partial | C++ gtest | `OMPBehaviour.after_fork` now checks the child exit status and passes directly. Focused C++ filters pass: engine/lifecycle `13`, DNNL/NDArray `8`, BatchNorm `16`, narrow operator basics `10`, and a quiet broad filter excluding topology, perf/timing, imperative, and DNNL sweeps passed `57/57`. Earlier broad C++ attempts were stopped for runtime/noise in `IMPERATIVE.CopyOp`/`IMPERATIVE.ActOp`, not visible assertions. | Schedule the remaining C++ suite in quieter shards, excluding known long topology cases and overly verbose imperative sweeps. |
| FS4 | Resolved | NCCL/multi-GPU | The old hard NCCL bandwidth threshold was load-sensitive and failed while other jobs were resident. It is now reported as a metric instead of a correctness assertion; the focused `test_nccl_bandwidth[1]` rerun passed. | Keep bandwidth values in logs for performance triage; do not fail correctness on this threshold. |
| FS5 | In progress | GPU miscellaneous | The full `tests/python/gpu` lane collected `13397` tests but aborted in `test_fu4_fork_safe_dnnl.py::test_dataloader_num_workers_4_no_primitive_failure`; that fork-path bug is fixed and the file now passes. A later broad non-operator GPU sweep was interrupted after partial results (`113 passed`, `9 failed`, `9 errors`) to avoid cascading OOM/noisy teardown failures. The failures clustered into leak-check teardown reports, deferred-compute global-state/order sensitivity, one DNNL fork-path CPU fp16 GEMM assertion, and high-memory float16 embedding checks; focused deferred-compute GPU rerun passed `31 passed`, and prior focused reruns for deconv TF32, reducer regressions, NumPy einsum GPU, cuBLASLt GEMM/FC/dtype/strided paths, fp16 batch-dot, linalg temp storage, KVStore local/device/GPU paths, NCCL multiprocess, and extension GPU behavior pass or skip cleanly. A mixed fast/regression shard initially showed order-sensitive deconv/device leak-check noise and a NumPy einsum failure, but the failed nodes pass alone; split reruns passed AMP/batchnorm/batch-dot/reducer/NumPy fallback/C5 fallback `33 passed`, deconv/device `5 passed`, TVM/linalg/extensions/transforms `13 passed, 3 skipped`, cuBLASLt GEMM/FC/dtype/strided `27 passed, 7 skipped`, and standalone NumPy einsum `1 passed`. The embedding NaN reproducer now defaults to a lightweight 4-case pytest sweep that passed, while the original large load is explicitly gated by `MXNET_EMBEDDING_NAN_STRESS=1` and still needs a scheduled stress run. | Continue smaller non-operator GPU shards, then isolate remaining leak-check/order-sensitive files and high-memory embedding behavior before treating broad GPU coverage as complete. |
| FS6 | In progress | GPU operator | The earlier full `tests/python/gpu/test_operator_gpu.py` lane now collects 12,984 items and was stopped previously at about 1% to reduce load; one early failure marker had appeared, but no summary was available. A bounded operator micro-shard covering convolution/deconvolution smoke, histogram, proposal, kernel-error, zero-size, reshape, and split cases now passes `26 passed, 12958 deselected`; the histogram-or-MultiProposal shard passed `6 passed, 12978 deselected` after CUDA checked-arithmetic hardening; the subgraph shard passed `342 passed, 12642 deselected` in 88.50s; the local conv/deconv option/type/version shard passed `8 passed, 12976 deselected` in 17.21s; the local pooling shard passed `8 passed, 12976 deselected` in 19.79s; the mixed local operator shard covering STN/grid/bilinear/concat/reshape/blockgrad/swapaxis/FC/activation/LRN/softmax/math/arange/sparse-device basics passed `517 passed, 12467 deselected` in 222.39s; the moderate shape/dtype/linalg-lite shard (`np_can_cast`, `np_cross`, `np_linalg_matrix_norm`, `np_linalg_vector_norm`) passed `928 passed, 12056 deselected` in 235.43s; the unique/transpose/broadcast shape shard (`np_unique`, `np_transpose`, `np_permute_dims`, `np_broadcast_to`, `np_broadcast_arrays`) passed `578 passed, 12406 deselected` in 135.33s; the elementwise/broadcast/slice shard (`np_standard_binary_funcs`, `np_standard_unary_funcs`, `np_max_min`, same-shape broadcast ops, `gamma_grad`, `npx_slice`) passed `459 passed, 12525 deselected` in 1194.81s; the imported NumPy sum shard passed `2016 passed, 10968 deselected` in 456.95s. | Continue sharding the full operator file rather than running it as a single monolithic process. |
| FS7 | Resolved | DNNL quantized conv+sum | Local mitigation disables residual conv+sum fusion during quantization, avoids channel-wise quantizing `_sg_onednn_conv` nodes that still carry `with_sum`, and fixes sum-input type restoration in DNNL conv inference. Focused rebuilt-library checks passed: `test_pos_conv_add3`, `test_conv_bn_sum`, `test_channelwise_quantize_model_skips_onednn_conv_with_sum`, and `test_quantize_gluon_with_forward`. | Keep these in the DNNL quantization subset for broad reruns. |

### Focused Test Gate Before Broad Reruns

- [x] DNNL quantized conv+sum: run
      `tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_pos_conv_add3`,
      `tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_conv_bn_sum`,
      `tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_channelwise_quantize_model_skips_onednn_conv_with_sum`,
      and
      `tests/python/dnnl/test_quantization_dnnl.py::test_quantize_gluon_with_forward`.
- [ ] DNNL quantization subset: rerun the Gluon quantization and oneDNN
      quantization files that exercise `_sg_onednn_conv`,
      `quantized_sg_onednn_conv`, and residual/add fusions.
- [ ] CPU unittest smoke: keep `tests/python/unittest/test_extensions.py` and
      the two prior Gluon GPU-memory-pressure nodes as focused checks before
      trusting another full `tests/python/unittest` summary.
- [ ] GPU smoke: keep the fork-safety DataLoader, cuBLASLt FC, TF32 deconv,
      cuDNN stream/workspace, deferred-compute GPU, extension GPU, and NCCL
      metric checks ahead of any monolithic `tests/python/gpu` rerun.
- [ ] C++ gtest: rerun the two BatchNorm mixed GPU/CPU filters directly before
      scheduling another broad `mxnet_unit_tests` pass.

---

## GitHub Delta Import

These rows were imported from the prior GitHub delta crawl so the main queue can
be processed without switching files constantly.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| GH1 | Open | Security/tooling | Open upstream security/tooling hardening includes unsafe extraction, command injection risk in notebook conversion, insecure links, and stale docs/CD dependencies. | Audit local scripts before using them on untrusted inputs; patch targeted risks rather than merging old PRs wholesale. |
| GH2 | Open | C/C++ inference API | C Predict and C++ inference/subgraph APIs have unresolved correctness gaps: duplicate input names, executor binding locking, deleted subgraph nodes, and API parity. | Schedule a CPU-reproducible inference API audit with regression tests. |
| GH3 | Open | Autograd/Gluon semantics | Unresolved gradient/export semantics include `grad_req='add'`, non-leaf `attach_grad`, `autograd.grad`, `block.export`, mixed binary backward, and dtype propagation. | Start with CPU tests, then add CUDA parity only after the runtime build is stable. |
| GH4 | Open | Resource hygiene | Logger/file-handle lifetime and repeated `Extract()` leaks are separate from the engine/DataLoader lifecycle work. | Add leak/file-handle regression tests around the affected utility paths. |
| GH5 | Open | Operator correctness | Stale operator PRs cover deformable-conv `slice_axis`, `linspace`, fp16 `argsort`, GroupNorm, NumPy inf reductions, and randomized ReLU. | Triage as focused CPU/GPU operator tests after the CUDA smoke batch. |
| GH6 | Open | Data/datasets | Dataset/RecordIO issues remain: multithreaded RecordIO file-id switching, ImageFolderDataset classes, transform handling, and MultiboxPrior coverage. | Fold into the data/image sweep after runtime stability is established. |
| GH7 | Deferred | Distributed training | Horovod KVStore lacks a barrier API, distinct from ps-lite/NCCL backlog. | Defer unless Horovod support is explicitly in scope. |
| GH8 | Deferred | Linux CPU performance | FlexiBLAS detection, transparent huge pages, and `parallel_for` grain tuning need Linux benchmarking. | Revisit after correctness and benchmark harnesses are in place. |
| GH9 | Deferred | TensorRT | TensorRT upgrade/build work is stale and separate from core CUDA validation. | Defer until CUDA CI exists. |

---

## Tracker Reconciliation Notes

These notes resolve stale or duplicate historical tracker entries against the
current source tree. `issues.md` is canonical for processing order.

| Source | Current interpretation |
|---|---|
| Removed `FOLLOW_UPS.md` FU-1 | The AVX2 int8 conv+relu tail gate is implemented and covered by `tests/python/dnnl/test_fu1_int8_ic_lt8_gate.py`; keep B1/B2 for broader Linux oneDNN INT8 validation. |
| Removed `FOLLOW_UPS.md` FU-2 / removed `github-issues.md` G10 | Mixed fp16/int8 quantization remains open, but B3 is the canonical row. |
| Removed `FOLLOW_UPS.md` FU-4 | Fork-safe oneDNN/DataLoader behavior is implemented; Linux validation belongs under L1/T11, not a fresh FU-4 investigation. |
| Removed `FOLLOW_UPS.md` FU-6 | QAT subgraph backward bodies are not present on current `master`; B4 is canonical. The old 2026-05-19 handover named a local-only branch `fix/fu6-qat-subgraph-backward`, commits `74b62dc27` and `a24610a02`, source `src/operator/subgraph/dnnl/dnnl_qat_backward.cc`, test `tests/python/dnnl/test_fu6_qat_subgraph_backward.py`, and env gate `MXNET_QAT_SUBGRAPH_BACKWARD=1`, but those refs/files are not present in the current clone. The handover's reported `13 PASS, 4 XFAIL` result was against an earlier WIP binary; it explicitly warned that tests had not been run on the fresh built binary. |
| Removed `FOLLOW_UPS.md` FU-8 | Its legacy A6/A7 labels refer to old engine-deadlock audit IDs, not current A6/A7 rows; lifecycle work is tracked by T11. |
| Removed `FOLLOW_UPS.md` FU-11 | Wide oneDNN stack/concat fallback is implemented and covered by `tests/python/dnnl/test_fu11_large_stack_concat.py`; D2L notebook-run/output-audit artifact intake is tracked under D3-D5/L4. |
| Removed root CUDA tracker markdown | `nccl_status.md`, `cudnn_autotune_v9.md`, `fp16_perf_bench.md`, `sparse_thrust3_bench.md`, `storage_pool_bench.md`, and `quantized_backward_status.md` were historical reports. Their active work is canonical under FS4/T3, C5/C6, C4, sparse/storage follow-up rows, and B4. |
| `issues.md` T12-T14 | These are resolved Apple/local oneDNN test-harness and fallback entries; the remaining work is Linux x86 oneDNN validation under L1/T11. |

---

## Active Apple Silicon / CPU Queue

These Apple Silicon findings are locally resolved, but the relevant shared
paths still need Linux x86/CUDA confirmation.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| A6 | Resolved locally | Resource shutdown | Custom-op workers and thread-local temp resources needed shutdown-order hardening. | Mirror the new lifecycle tests on Linux x86/CUDA before calling this platform-complete. |
| A7 | Resolved locally | macOS multiprocessing | Some Apple Silicon/macOS environments allow Python `multiprocessing.shared_memory` probes to pass while MXNet `cpu_shared` allocation fails with `shm_open: Operation not permitted`, breaking multiworker `DataLoader`. | Validate the pickle-transport fallback on Linux x86/CUDA; it should remain inactive there when `cpu_shared` works. |
| A8 | Resolved locally | macOS wheel | Build a slim optimized Apple Silicon wheel with Accelerate, oneDNN, OpenMP, OpenCV, and libjpeg-turbo, but without ONNX/MPS/GPU. | Install and smoke-test the wheel on another Apple Silicon machine if available; then move to Linux/CUDA validation. |

### Resolved On This Follow-Up Branch

| ID | Area | Resolution | Verification |
|---|---|---|---|
| R1 | Python DataLoader cleanup | Iterator `close()`, abandoned-pool retirement, timeout cleanup, and thread-pool batchify fix. Commit `342a8ab20`. | Focused `test_gluon_data.py` worker/leak tests passed. |
| R2 | Image short-buffer reads | JPEG/PNG sniffers now length-check before reading headers. Commit `4604260fe`. | `test_image.py` and focused malformed-header tests passed. |
| R3 | libjpeg-turbo RecordIO cleanup | Malformed JPEG paths use full encoded byte size and RAII for `tjhandle`. Commit `4604260fe`. | Focused RecordIO image tests passed. |
| R4 | Histogram CPU validation | Reject non-positive uniform bin counts; avoid explicit-bin right-edge overread. Commit `513382825`. | Focused CPU histogram tests passed. CUDA parity is deferred. |
| R5 | Fixed-capacity fused optimizers | `multi_all_finite`, fused SGD, preloaded SGD/LARS, AdamW, AdaBelief, LAMB, and LANS validate grouped counts before filling fixed launch structs. Commit `58f5befad`. | 17 focused overflow tests passed. |
| R6 | Quantized flatten empty tensors | Quantized min/max outputs are initialized even for empty flattened data. Commit `513382825`. | Focused CPU regression passed. |
| R7 | Proposal CPU sizing | CPU `Proposal` and `MultiProposal` use checked 64-bit arithmetic before narrowing to `mshadow::index_t`. Commit `e04b40750`. | Incremental rebuild and `test_multi_proposal_op` passed. CUDA variants are deferred. |
| R8 | oneDNN quantized transpose | Scalar min/max range outputs are honored independently of data-output requests. Commit `db134fed4`. | Focused oneDNN regression passed. |
| R9 | Azure option | `USE_AZURE=ON` fails clearly at configure time instead of compiling incomplete dmlc-core Azure support. Commit `c4ffce3c2`. | Configure-failure check passed. |
| R10 | oneDNN generated headers | Removed post-build copying of generated oneDNN headers into the source tree. Commit `c4ffce3c2`. | oneDNN configure/generate check passed. |
| R11 | Plugin unload ownership | Python plugin handles are retained for process lifetime because MXNet has process-lifetime plugin registries and no unregister API. Commit `c4ffce3c2`. | Extension load tests passed. |
| R12 | DataLoader | C++ no-python DataLoader resets in a generator `finally` block, so early close/break returns the iterator to the first batch. | `test_mx_data_loader_nopython_early_close_resets` passed. |
| R13 | OpenMP | `src/engine/openmp.*` shared state now uses atomics instead of `volatile`. | `OMPBehaviour.concurrent_state_access` passed. |
| R14 | Custom operators | Custom-op async exceptions are stored per queued invocation instead of in a singleton. | `test_custom_op_exception_isolation_between_queued_ops` and `test_custom_op_exc` passed. |
| R15 | KVStore CPU | `CommCPU::Reduce` async lambdas capture arrays/resources and scalar config by value, not `this`. | `test_local_kvstore_delete_before_wait_releases_async_reduce` passed. |
| R16 | Threaded engine | ThreadedEngine global and per-var exception refs are guarded by an exception mutex and cleared consistently. | `Engine.ThreadedAsyncExceptionsAreReportedOnce` passed. |
| R17 | oneDNN C++ tests | C++ oneDNN unit-test helpers use current oneDNN descriptor APIs instead of stale `memory::desc.data` / `convolution_forward::desc` access. | `mxnet_unit_tests` built; `DNNL_UTIL_FUNC.*` and `DNNL_NDArray.GetDataReorder` passed. |
| R18 | Lifecycle test coverage | Added focused regressions for DataLoader worker exceptions, KVStore row-sparse delete-before-wait, custom-op backward exception isolation, and `WaitForVar` exception clearing. | Focused Python lifecycle sweep and `Engine.WaitForVarClearsThreadedAsyncException` passed. |
| R19 | Engine shutdown | Threaded engine start/stop is idempotent, late `DeleteVariable` work is drained safely, and custom-op worker teardown reports queued errors per invocation. | `test_engine_shutdown.py` plus the focused shutdown/DLPack/autograd/quantization sweep passed. |
| R20 | DLPack CPU interop | Incoming DLTensor `byte_offset` is honored and Python capsule validation raises explicit `ValueError`s instead of relying on asserts. | `test_data_interchange.py` CPU DLPack tests and `test_dlpack_from_nonzero_byte_offset` passed. |
| R21 | NumPy API drift | Removed noisy/stale aliases such as `np.Inf`, `np.NINF`, `np.NaN`, `np.PZERO`, `np.NZERO`, and `np.product` from tested paths. | Focused NumPy/DLPack/autograd sweep passed under the current NumPy 1.x test environment. |
| R22 | Apple Silicon oneDNN fallbacks | AArch64 oneDNN paths that hit Xbyak internal errors now fall back for quantized ops, batch-dot, transpose/reorder, RNN, scalar pow/mul, and default-layout reshape. | Direct Xbyak repros for quantize, batch-dot, transpose, and RNN no longer crash; native quantization suite passed. |
| R23 | Quantized transpose/requantize CPU fallback | `requantize` has a native CPU registration in oneDNN builds, and quantized transpose has a native CPU implementation that preserves range outputs. | `tests/python/quantization/test_quantization.py` passed except the expected AArch64 oneDNN quantized-RNN skips. |
| R24 | C++ oneDNN pooling tests | Pooling test helpers now derive forward/backward arity from the parsed operator parameters instead of stale input-dimensionality assumptions. | `mxnet_unit_tests --gtest_filter=IMPERATIVE.PoolingOp` passed. |
| R25 | C++ oneDNN convolution tests | The oneDNN-vs-native convolution fixture now compares floating outputs with numeric tolerances instead of raw `memcmp`, preserving the existing data-gradient coverage. | `mxnet_unit_tests --gtest_filter=IMPERATIVE.ConvOp` passed. |
| R26 | C++ BatchNorm stochastic test | BatchNorm validators now count samples per channel so single-sample stochastic normalization groups are not incorrectly checked for unit variance. | `mxnet_unit_tests --gtest_filter=BATCH_NORM.TestStochasticTiming_2D --gtest_repeat=20` passed. |
| R27 | Apple Silicon oneDNN float fallbacks | AArch64 oneDNN JIT-backed float primitives now fall back for activation, leaky ReLU, pooling, convolution, deconvolution, softmax/log-softmax, softmax-output, batch norm, dot, batch-dot, NumPy binary broadcast, sum, concat/stack/split, eltwise, layer norm, and where. Matching oneDNN graph rewrites are disabled on AArch64 when they would bypass these operator-level gates. | Expanded `test_apple_silicon_onednn_fallback.py` passed; ResNet-18 forward passed after the NumPy binary-add guard; the full previous Python failure replay passed except sandboxed POSIX shared-memory skips; full `mxnet_unit_tests` passed 89/89 after the direct oneDNN helper fixes. |
| R28 | Gluon model-zoo NumPy semantics | `test_gluon_model_zoo.py` now scopes `mx.npx.reset_np()` with an autouse fixture and restores the previous NumPy semantics after each test. | `test_models[resnet18_v1]` plus `test_recordimage_dataset` passed. |
| R29 | C++ stochastic shape helper | `rangedRand(min,max)` now samples the inclusive `[min,max]` interval instead of treating the range as `[min,max+1]` whenever `min != 0`. | `mxnet_unit_tests --gtest_filter=BATCH_NORM.TestStochasticTiming_2D --gtest_repeat=100` passed. |
| R30 | Apple Silicon smoke manifest | Added lifecycle, DLPack byte-offset, DataLoader, and oneDNN AArch64 fallback checks to `tests/python/apple_silicon_cpu_smoke`. | Listed Python checks passed; process-worker DataLoader checks now skip only when POSIX shared memory is unavailable. |
| R31 | C++ oneDNN AArch64 helpers | C++ oneDNN fixtures now avoid synthetic blocked layouts on AArch64, and direct oneDNN memory copy/sum helpers use contiguous CPU fallbacks for plain buffers instead of Xbyak-backed reorder/sum primitives. | `IMPERATIVE.ConcatBackwardsOp`, `DNNL_NDArray.CopyFrom`, `DNNL_BASE.DNNLMemorySum`, and `DNNL_BASE.CreateDNNLMem` passed; full C++ sweep passed 89/89. |
| R32 | Stale XPASS markers | The Windows-only `slogdet` xfail no longer matches `darwin`, and the stale boolean-index assignment xfail was removed. | `test_np_linalg_slogdet` passed 84/84 parameters and `test_boolean_index_assign` passed. |
| R33 | Random binomial test | The binomial-generator statistical test now accounts for output dtype quantization before computing expected decile buckets. | Full optimized Python sweep found this as the only pre-existing non-DataLoader failure; focused `test_random.py` passed after the fix. |
| R34 | Apple Silicon oneDNN JIT fallbacks | AArch64 oneDNN/Xbyak fallback gates were expanded and hardened for the current optimized build profile. | Optimized C++ suite passed 89/89; focused AArch64 oneDNN fallback and quantization checks passed before the final wheel build. |
| R35 | DataLoader `cpu_shared` fallback | Multiworker DataLoader now probes actual MXNet `cpu_shared` allocation and, if unavailable, keeps process workers but uses normal pickle transport instead of shared-memory NDArray transport. | `test_recordimage_dataset_with_data_loader_multiworker` and `test_multi_worker_falls_back_to_pickle_transport_without_cpu_shared` passed against the optimized library. |
| R36 | ONNX-free wheel packaging | `MXNET_SETUP_EXCLUDE_ONNX=1` excludes `mxnet.onnx` and `mxnet.contrib.onnx` packages/data; `mxnet.contrib` tolerates their absence; staged runtime libraries under `mxnet/lib` are packaged. | Fresh-venv wheel smoke test confirmed `mxnet.onnx` and `mxnet.contrib.onnx` are absent, `OPENMP/OPENCV/ONEDNN` are enabled, and a basic NDArray op works. |
| R37 | Linux oneDNN batch-dot | oneDNN batch-dot now reorders MXNet buffers into the primitive-selected descriptors before execution, including the subgraph temp-space request needed for `_sg_onednn_batch_dot`. Commit `4b54ccf8d`; coverage strengthened in `a460f4cd3`. | DNNL subgraph/regression shard passed `1005 passed, 16 skipped, 4 xfailed`; transformer standalone repro passes with oneDNN enabled. |
| R38 | Linux BF16 tests | Native BF16 DNNL tests now skip when the CPU lacks BF16 instructions, matching this Zen 2 host. Commit `d2c2c1beb`. | `tests/python/dnnl/test_amp.py` passed `45 passed, 34 skipped`. |
| R39 | GPU pytest harness | Pytest interruption handling no longer assumes `rep_call` exists, cublasLt FC test inputs use the NumPy NDArray boundary expected by Gluon Dense, and NCCL bandwidth is a metric rather than a hard correctness gate. Commit `37089f2e8`. | `test_cublaslt_fc.py` passed `9 passed, 3 skipped`; focused NCCL bandwidth rerun passed. |
| R40 | Linux DataLoader fork path | The DNNL fork-safety DataLoader test now forces the Python worker path it intended to exercise instead of silently taking the no-python loader. Commit `6b36b3c13`. | `tests/python/gpu/test_fu4_fork_safe_dnnl.py` passed `3 passed`; the prior full-GPU abort site is fixed. |
| R41 | Linux wheel OpenCV guard | Runtime bundling now refuses to silently leave OpenCV SONAME dependencies to the host unless OpenCV is bundled, already staged, or explicitly allowed, setup metadata now declares `opencv-python>=4,<5` when OpenCV support is enabled, and the Linux release-wheel workflow explicitly builds OpenCV-off preview wheels. Commits `c429a5207` and `d41d3cf1c`; local release-workflow follow-up is pending commit. | `tests/python/unittest/test_wheel_runtime_packaging.py` passed `7 passed`; wheel rebuild still needs final runtime-dependency audit. |
| R42 | Linux OpenMP fork test | `OMPBehaviour.after_fork` now checks the child process exit status so fork regressions cannot pass when the child fails independently. Commit `b881a9c5a`. | `build/tests/mxnet_unit_tests --gtest_filter=OMPBehaviour.after_fork` passed. |
| R43 | Optional extension artifacts | CPU/GPU extension tests now skip absent optional shared libraries instead of failing before runtime behavior can be tested. Commit `ac708930d`. | CPU extension file passed `4 passed, 1 skipped`; GPU extension file reported `2 skipped` because optional GPU/external artifacts are not built. |
| R44 | DNNL adaptive-pooling timeout | The expensive adaptive-pooling numeric-gradient test no longer runs the full row-sparse/default cross product; it keeps one small row-sparse case and a small default-storage matrix. Commit `119002dc9`. | `tests/python/dnnl/test_dnnl.py::test_adaptive_pooling` passed `5 passed`. |
| R45 | dmlc optional lifetime | `dmlc::optional` no longer swaps inactive raw storage or assigns failed stream extractions. dmlc-core commit `d610d79`; main submodule pointer recorded in `7816eb2e4`. | Standalone dmlc optional smoke compiled and ran; the configured dmlc CTest target currently reports no registered tests. |
| R46 | Product reducer warning cleanup | Product/nanprod reducers now initialize unused residual storage and use logical AND for bool reductions on CPU and CUDA RTC paths. Commit `7816eb2e4`. | Rebuilt `mxnet` with no pending work; `test_reducer_regressions.py` passed on CPU and GPU, and `test_np_prod` plus `test_device_pushpull` passed. |
| R47 | Compiler warning cleanup batch | Current local batch addresses CN4/CN5/CN6/CN7/CN8 warning clusters across reducer residuals, unsigned negative checks, local sentinel vectors, half parameter packing, KVStore NCCL naming, pointwise fusion initialization, and unused variables. | `build/libmxnet.so` rebuilt at 2026-05-21 07:47:21 UTC. Focused reducer, packaging, optimizer, DNNL conv/sum, and forward-only NumPy checks passed; GPU/KVStore/autograd-covered checks are deferred under L9. |
| R48 | Compiler warning cleanup follow-up | Current local batch addresses the remaining repo-owned CN8 warning clusters: mshadow packet allocation initialization, CTC/moderngpu include-boundary noise, einsum initialization, half CPU max-pooling negative infinity, and D2L tracker ownership cleanup. | `build/libmxnet.so` rebuilt at 2026-05-21 14:47:01 UTC after cleaning submodule-local experiments. Warning scan shows only CN2/CN9 boundary clusters remain. Focused CTC, CPU/GPU einsum, CPU/GPU pooling smoke, GPU reducer/TF32 deconv, reducer/packaging/optimizer tests passed. |

---

## Linux/CUDA Validation Backlog

These are real findings that need validation on the current Linux/CUDA host.
Use targeted repros before launching full GPU sweeps.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| C1 | Resolved | Histogram CUDA | CPU histogram validation fixes now have CUDA parity coverage: existing GPU histogram and NumPy histogram nodes pass, and `test_histogram_gpu_edge_and_invalid_bins` covers the right-edge and invalid-bin cases directly on GPU. | Keep the focused histogram GPU nodes in the CUDA smoke shard. |
| C2 | Resolved | Proposal CUDA | The CUDA `Proposal` and `MultiProposal` paths now use checked arithmetic for shape products, NMS mask sizing, allocation byte counts, host-vector sizes, pointer offsets, and kernel `int` launch counts. The rebuilt GPU `test_multi_proposal_op`, the CPU `test_multi_proposal_op`, and the histogram-or-MultiProposal GPU shard pass. | Keep `test_multi_proposal_op` and the histogram/proposal shard in CUDA smoke coverage; no active Proposal-specific overflow/narrowing work remains. |
| C3 | Resolved | cuBLASLt | Direct cuBLASLt GEMM, FC, dtype, and strided tests pass together: `27 passed, 7 skipped`. A same-process threaded stress using 2 threads per GPU across all 4 RTX 4090s passed dot and strided batch-dot loops with mixed fp32/fp16 work. A prior GPU1 environment-prefixed rerun failed with CUDA 304 in subprocess children, matching the known restricted command-context artifact and not a runtime regression. | Keep the direct cuBLASLt subset and same-process threaded stress pattern in CUDA smoke/performance triage; keep CUDA 304 environment-prefix failures out of correctness accounting. |
| C4 | Open | CUDA build matrix | Ada/Hopper/Blackwell architecture coverage and older CUDA 12.x compatibility need CI coverage. | Validate `sm_89` here; leave CUDA 12.x and dedicated Blackwell to later runners. |
| C5 | Resolved | cuDNN frontend | Representative cuDNN convolution/deconvolution smoke now passes on Ada, the higher-risk large-channel probes `test_convolution_large_c` and `test_deconvolution_large_c` pass, frontend plan selection works on a large-C NCW convolution including a zero-workspace variant, and explicit no-plan fallback is now covered by `MXNET_CUDNN_FORCE_NO_HEURISTIC_PLANS=1` in `tests/python/gpu/test_cudnn_frontend_no_plan_fallback.py`. The fallback builder also skips unsupported engine-config finalization statuses instead of aborting before trying later fallback engines. | Keep frontend-autotune and no-plan fallback tests in CUDA coverage; reopen only for a fresh cuDNN frontend/fallback runtime failure. |
| C6 | Resolved | cuDNN streams | The old skipped multi-stream regression compared cuDNN against a weak cross-backend training path. It now runs cuDNN directly under `MXNET_GPU_WORKER_NSTREAMS=1/2` and `NaiveEngine`, `ThreadedEngine`, and `ThreadedEnginePerDevice`, using deterministic inputs and output gradients with a NumPy oracle for forward, data-gradient, weight-gradient, and bias-gradient results. The focused node passes locally with the adjacent conv/deconv guard. | Keep `test_convolution_multiple_streams` in CUDA smoke coverage; if future failures appear, treat them as cuDNN wrapper behavior against the independent oracle rather than as non-cuDNN numeric drift. |
| C7 | Resolved | CUDA kernels | The targeted CUDA edge shard now passes split, reshape, reducer, kernel-error, zero-size tensor, concat, and `zero_sized_dim` coverage through the 26-node bounded GPU operator micro-shard plus `tests/python/gpu/test_reducer_regressions.py`. | Keep the micro-shard in CUDA smoke coverage; broader operator completeness remains tracked under FS6/T1. |
| C8 | Resolved | TF32 deconvolution | Tracker was stale: `cudnn_deconvolution-inl.h` now mirrors the convolution TF32 guard and `tests/python/gpu/test_deconv_tf32.py` exists. The Ada rerun passed `4 passed`. | Keep the focused test in CUDA smoke coverage; no current TF32 deconvolution fix is needed. |

---

## Blackwell / CUDA Correctness Backlog

These are older Linux/CUDA findings from the Blackwell port. They are not the
current Apple Silicon task, but they still matter for the fork.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| B1 | Resolved | oneDNN INT8 subgraphs | The current Linux x86 oneDNN build now passes the full INT8 matmul/self-attention subgraph file: `tests/python/dnnl/subgraphs/test_matmul_subgraph.py` reported `64 passed`. The earlier `test_self_attention[*]`, `test_batch_dot[*]`, and `test_self_attention_negative` blow-ups no longer reproduce on this rebuilt library. | Keep the matmul subgraph file in the oneDNN validation subset; reopen only for a fresh INT8 subgraph runtime or accuracy failure. |
| B2 | Resolved | Quantized Gluon | The historical `test_quantize_gluon_with_forward` segfault no longer reproduces, and the full DNNL quantization file now passes on Linux x86: `tests/python/dnnl/test_quantization_dnnl.py` reported `26 passed`. The related conv+residual-sum quantization checks under FS7 also pass. | Keep `test_quantization_dnnl.py` and the FS7 focused conv+sum checks in oneDNN validation; reopen only for a fresh quantized-Gluon or DNNL quantization failure. |
| B3 | Resolved | Mixed dtype quantization | True fp16 quantize/dequantize kernels remain unsupported by design, but AMP now treats `_contrib_quantize_v2` and `_contrib_dequantize` as FP32 boundary ops instead of dtype-neutral fp16/fp32 ops. Focused tests cover the AMP cast boundary, dequantize output staying fp32, direct float16 `quantize_v2` rejection, and AMP coverage now includes `_contrib_quantized_npi_add`. | Documented behavior is fp32 at AMP quantization boundaries; reopen only if native fp16 quantization support is explicitly required later. |
| B4 | Partial | QAT backward | STE for `quantize_v2` and QAT parameter `grad_req` fixes landed. The 2026-05-19 handover reported a local-only gated DNNL subgraph backward implementation on branch `fix/fu6-qat-subgraph-backward` with commits `74b62dc27` and `a24610a02`, env gate `MXNET_QAT_SUBGRAPH_BACKWARD=1`, source `src/operator/subgraph/dnnl/dnnl_qat_backward.cc`, and test `tests/python/dnnl/test_fu6_qat_subgraph_backward.py`; none of those refs/files are present in this clone. The handover's `13 PASS, 4 XFAIL` note was from an earlier WIP binary, not a fresh rebuilt-library validation. | Recover that local branch from another clone/reflog if possible, otherwise redesign the gated `_backward_sg_onednn_*` implementation from the handover notes and validate scale magnitude before PR. |
| B5 | Open | Mixed dtype matrix coverage | fp16/fp32 AMP, int8/fp32 quantize, and int8/fp16 combinations are separate paths. | Build an explicit coverage matrix; do not infer one path from another. |

---

## Performance / Nonblocking Engineering

These should not block basic correctness on Apple Silicon, but they affect the
quality of a public fork.

| ID | Status | Area | Issue |
|---|---|---|---|
| P1 | Deferred | cuBLASLt | fp32/fp16/bf16/fp64 env-gated cuBLASLt paths landed, but default-on, stride-aware, and INT8 follow-ups remain. |
| P2 | Resolved | TF32 | Duplicate of C8; keep the actionable CUDA validation under C8 and do not track a second TF32 deconvolution row here. |
| P3 | Deferred | Sparse/topk | `topk(k=10/100/1000)` is K-independent because MXNet sorts full rows then slices. |
| P4 | Deferred | Small ops | Softmax, LayerNorm, and small elementwise ops are slower than PyTorch due to multi-pass kernels and dispatch overhead. |
| P5 | Hardware | bf16 CPU | The tested Zen 2 host lacks AVX-512 BF16; oneDNN falls back to fp32 emulation. Validate on BF16-capable CPUs. |
| P6 | Resolved | Storage pool default | Round pool used more memory than Naive on ResNet-18; keep the default unless a workload proves otherwise. |

---

## Test Coverage And Integrations

| ID | Status | Area | Issue |
|---|---|---|---|
| T1 | Open | GPU tests | A full `test_operator_gpu.py` sweep is still incomplete on the current Linux/Ada host. The host driver mismatch is resolved, focused GPU smoke/regression tests pass, and a bounded 26-node operator micro-shard passes, but the broad operator lane still needs systematic sharding. |
| T2 | Open | Out-of-tree users | GluonNLP, Sockeye, AutoGluon, and DGL compatibility is untested. DGL is explicitly out of scope for the current Apple Silicon work. |
| T3 | Partial | Distributed training | Single-machine local/device/NCCL tests passed on Blackwell; multi-machine ps-lite rendezvous was not deployed. |
| T4 | Open | Python versions | Python 3.13+ is untested. |
| T5 | Partial | NumPy ABI | NumPy 2.x compatibility is not established; several failures already came from API drift. |
| T6 | Partial | DLPack | CPU byte-offset handling and NumPy interchange are fixed; PyTorch/JAX and CUDA interop remain stale/untested. |
| T7 | Resolved | Data/image tests | Earlier `test_gluon_data.py`, `test_contrib_gluon_data_vision.py`, and `test_image.py` crashes no longer reproduce in isolation after data/batchify fixes. |
| T8 | Resolved | ONNX opset 18 reductions | Exporter now emits axes as input tensors for opset >=18. Broad suite had one unrelated fp16 softmax numerical failure. |
| T9 | Resolved | Gluon model zoo | 34/34 model zoo checks passed, aside from a pre-existing `test_parallel_download` skip. |
| T10 | Resolved | Custom C++ operators | 9/9 in-tree extension/custom-op checks passed on Blackwell. |
| T11 | Open | Cross-platform lifecycle coverage | Apple Silicon now has focused async lifecycle tests. Mirror and validate the same patterns on Linux x86 CPU/oneDNN and CUDA before treating them as platform-complete. |
| T12 | Resolved locally | C++ oneDNN pooling | Full `mxnet_unit_tests` on Apple Silicon reached `IMPERATIVE.PoolingOp` failure: `outputs.size() == GetNumOutputs(param) (1 vs. 2)`. | Fixed by deriving fixture arity from parsed pooling params; focused Apple Silicon run passed. Validate in Linux x86 oneDNN CI. |
| T13 | Resolved locally | C++ oneDNN convolution | Full `mxnet_unit_tests` on Apple Silicon reached `IMPERATIVE.ConvOp` oneDNN-vs-native data mismatches in `tests/cpp/include/test_dnnl.h:637`. | Fixed as a test-harness comparison issue by replacing raw `memcmp` with tolerant numeric comparison; focused Apple Silicon run passed. Validate in Linux x86 oneDNN CI. |
| T14 | Resolved locally | Apple Silicon oneDNN fallbacks | Fresh-process Python model and operator tests still found AArch64 Xbyak crashes outside the earlier quantized/RNN scope. | Added fallback coverage for the remaining failing float primitive families and a fresh-process regression test. Validate on Linux x86 oneDNN CI to ensure those paths remain enabled there. |
| T15 | Resolved locally | Optimized Apple Silicon C++ sweep | Optimized `-O3 -DNDEBUG -g0 -mcpu=apple-m1` C++ test build completed with OpenMP, OpenCV, oneDNN, Accelerate, and libjpeg-turbo enabled. | `mxnet_unit_tests` passed 89/89 in `build-macos-arm64-slim-optimized`. |
| T16 | Resolved locally | Optimized Apple Silicon Python sweep | Full optimized Python unittest sweep was run before the final DataLoader fallback. It finished with exactly one failure, the known macOS `cpu_shared` DataLoader failure that was fixed afterward. | Pre-fix result: `1 failed, 14045 passed, 67 skipped, 65347 warnings in 1:26:34`; post-fix targeted DataLoader checks passed. Full re-run was skipped because the final fix is Python-only and targeted coverage passed. |
| T17 | Resolved locally | macOS wheel smoke | Final ONNX-free macOS arm64 wheel was installed into a fresh UV-created venv. | Import succeeded; `mx.__version__ == 2.0.0+macos.arm64.20260520`; ONNX specs were absent; `OPENMP`, `OPENCV`, and `ONEDNN` reported enabled; `mx.nd.ones((2,3)).sum()` returned `6.0`. |

### Cross-Platform Lifecycle Coverage TODO

These are the follow-ups for mirroring Apple Silicon lifecycle coverage across
the rest of the build matrix:

- [ ] Linux x86 CPU: run DataLoader, ThreadedEngine, KVStore, and custom-op
      lifecycle tests with and without oneDNN.
- [ ] Linux CUDA: add or enable CUDA analogues for engine exception propagation,
      KVStore lifetime, and custom-op forward/backward failure isolation.
- [ ] Linux CUDA: run the same lifecycle tests with `NaiveEngine`,
      `ThreadedEnginePooled`, and `ThreadedEnginePerDevice` where supported.
- [ ] Sanitizers: run the C++ engine/OpenMP/KVStore subset under TSAN, and the
      C++/Python lifecycle subset under ASAN/UBSAN where the toolchain allows.
- [ ] CI: add a quick job that builds `mxnet_unit_tests` and runs the focused
      lifecycle filters before any expensive full-suite job.

---

## Build, Release, And Operations

| ID | Status | Area | Issue |
|---|---|---|---|
| O1 | Open | Wheel packaging | Linux wheel does not bundle CUDA/cuDNN/NCCL runtimes; users need system packages. |
| O2 | Open | CUDA arch matrix | The old public Blackwell wheel was effectively sm_120-only; useful public wheels need sm_80/sm_86/sm_89/sm_90/sm_120 coverage. The source CMake default now lists these, but release builds still need verification. |
| O3 | Open | CI | `smolix/mxnet` still lacks CI. Even a small build plus DNNL subset would catch regressions. |
| O4 | Open | Release | GitHub Release and wheel publication are not automated. The public GitHub release `v2.0.0+cu13.bw.20260518.2` was published on 2026-05-18 at 20:30:11 UTC and predates later master fixes from the 2026-05-19 handover, so do not treat that wheel as current-source provenance. |
| O5 | Open | Changelog | No clear release notes for CUDA 13, cuDNN 9, oneDNN v3, quantization, and Apple Silicon changes. |
| O6 | Open | README/docs | README and docs still read like archived Apache MXNet and omit modern dependency guidance. |
| O7 | Open | Packaging ecosystem | No conda/system package story for this fork. |
| O8 | Strategic | Upstream status | Apache MXNet was archived on 2023-11-17; all future fixes must live in this fork or downstream users must migrate. |
| O9 | Strategic | oneDNN cadence | Future oneDNN major releases will likely require repeated porting work. |
| O10 | Resolved locally | macOS wheel artifact | Slim optimized CPython 3.12 macOS arm64 wheel built with `-mcpu=apple-m1`, Accelerate, oneDNN, OpenMP, OpenCV, and libjpeg-turbo; ONNX and MPS/GPU are excluded. | Artifact: `dist/mxnet-2.0.0+macos.arm64.20260520-cp312-cp312-macosx_11_0_arm64.whl`; SHA256 `3953e9ad44934259ab0518f2c00f29bd0bd7bff8d959c1093fd1d3c2371a20af`. The `dist/` directory is intentionally not part of the PR. |
| O11 | Open | Variant versioning | Source `python/mxnet/libinfo.py` still carries the prior CUDA local version suffix. The macOS wheel was stamped in the staging tree only to avoid changing CUDA packaging metadata on this branch. The local editable package reports `2.0.0+cu13.bw.20260518.1`, while the native library still reports configure-time commit `b881a9c5a` even after later commits and rebuilds, because CMake metadata was not regenerated. The public CUDA wheel tag also lagged source fixes during the 2026-05-19 handover, reinforcing that local-version tags must identify the exact source/build payload. | Decide a general versioning/provenance scheme for CPU/CUDA/macOS wheels before public release automation. Generate package/build metadata from an explicit release version or git description, include commit/dirty state/build options/submodule SHAs in the wheel, and reconfigure before release builds. |

---

## Resolved Historical Highlights

These were major Blackwell/CUDA port findings that are now fixed or documented.
They stay here as context, not as active work.

| Area | Outcome |
|---|---|
| Adaptive average pooling | DNNL adaptive avg-pool backward disabled in favor of correct CPU fallback; 72/72 adaptive-pool checks passed. |
| Quantize asym | oneDNN v3 attr-on-reorder issue fixed with `set_scales_mask(DNNL_ARG_DST, 0)`. |
| INT8 conv concat/relu/u8 | Runtime and property gates avoid the oneDNN small-channel u8 post-op bug; full conv subgraph passed after the fix. |
| Softrelu backward | `test_activation` softrelu issue resolved and unskipped. |
| Random seeding | CPU random generators are per logical CPU dev_id, matching GPU behavior. |
| fp16 batch dot | Batched fp16 GEMM now uses fp32 accumulation via `cublasGemmStridedBatchedEx`; parity tests passed. |
| CUDA linalg temp storage | Ephemeral GPU scratch now synchronizes before free; linalg stress tests passed. |
| AMP subgraph | BF16-on-AVX2 fallback upcasts to fp32 for unsupported oneDNN primitives. |
| AMP RNN conversion | Test was repaired and passed on cuDNN 9; the original upstream waitall failure did not reproduce. |
| NCCL single-process | Single-process 2-GPU NCCL KVStore tests passed; multi-process DDP-style NCCL is outside MXNet KVStore design. |
| Test-source bugs | Several stale numpy/op tests were fixed or correctly skipped, freeing many blocked test invocations. |
| GPU profiler symbolic test | oneDNN v3 node-name expectation updated; test passed. |
| ONNX | Opset 18 reduction API change handled. |
| cuDNN 9.22 bump | Depthwise conv performance improved substantially on Blackwell; smoke tests stayed clean. |
| cuDNN frontend autotune | Env-gated v9 frontend autotune path added; default remains conservative. |
| sm_120 SASS | Confirmed `12.0+PTX` already emitted sm_120 SASS; SASS-only rebuild can shrink artifacts later. |
| Sparse ops | CUDA 13 / Thrust 3 sparse benchmarks showed no port regression. |
| fp16 tensor cores | Large dense/conv fp16 tensor-core paths are near PyTorch parity on Blackwell. |

---

## Suggested Triage Order

Current Linux/Ada host:

1. Use the completed L0/L2 evidence as the current validation baseline.
2. Complete L1 and L3: validate shared Apple Silicon fixes on Linux x86, then
   run targeted CUDA regressions across all 4 GPUs where useful.
3. Consume current D2L notebook-run/output-audit artifacts now that the
   standalone GPU and transformer runtime probes are clean; reopen MXNet work
   only for fresh runtime repros.
4. Work through C1-C8, B1-B5, and T11 using focused tests before full sweeps.

Before shipping another public Linux/CUDA preview wheel:

1. Re-test B1/B2 on the newest master.
2. Run T1 full GPU operator sweep.
3. Resolve or document C1-C8 and D1; keep D2-D4 resolved unless fresh repros
   arrive, keep D5-D7 external, and keep D8 informational unless current
   artifacts produce a concrete MXNet runtime repro.
4. Add at least minimal CI, including a GPU job once a runner is available.
5. Update README, dependency docs, release notes, and wheel/versioning policy.

Pointers:

- `issues.md` is the canonical tracker. Stale root CUDA tracker markdown has
  been removed after import; use git history for historical report details.
