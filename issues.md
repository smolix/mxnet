# MXNet Port Issues

Updated: 2026-05-20
Current branch: `followup/full-sweep-macos-wheel`
Code baseline when reorganized: `e04b407507`; current follow-up base: `862669419`

This file is a status index, not a changelog. Historical details live in git
commits, `handover.md`, and the linked investigation notes. Items are grouped
by what a maintainer needs to decide next.

Status labels:

- **Open**: known issue; no verified fix on the current branch.
- **In progress**: local work exists but is not committed and verified yet.
- **Deferred**: cannot be verified on this Apple Silicon machine, or is not on
  the current Apple Silicon path.
- **Resolved**: fix is committed or otherwise verified; retained here only when
  it is useful context for future work.

---

## Active Apple Silicon / CPU Queue

These are the remaining non-CUDA findings from the Apple Silicon bring-up audit.
They can be fixed or at least partially verified on this Mac.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| A6 | Resolved locally | Resource shutdown | Custom-op workers and thread-local temp resources needed shutdown-order hardening. | Mirror the new lifecycle tests on Linux x86/CUDA before calling this platform-complete. |

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

---

## Deferred CUDA / Linux Validation

These are real findings, but this Mac cannot validate CUDA behavior. Do not
change them blindly on Apple Silicon unless the fix is clearly shared CPU code.

| ID | Status | Area | Issue |
|---|---|---|---|
| C1 | Deferred | Histogram CUDA | Mirror or validate the CPU histogram fixes in CUDA kernels. |
| C2 | Deferred | Proposal CUDA | Audit CUDA `Proposal` / `MultiProposal` for the same overflow and narrowing risks fixed on CPU. |
| C3 | Deferred | cuBLASLt | Shared workspace lifetime/race risk needs Linux/CUDA stress testing. |
| C4 | Deferred | CUDA build matrix | CUDA architecture defaults and older CUDA 12.x compatibility need CI coverage. |
| C5 | Deferred | cuDNN frontend | No-plan frontend autotune paths should fall back instead of aborting. |
| C6 | Deferred | cuDNN streams | The skipped multi-stream regression still needs CUDA validation. |
| C7 | Deferred | CUDA kernels | Zero-block launches and GPU split edge cases need targeted CUDA tests. |
| C8 | Deferred | TF32 deconvolution | `cudnn_deconvolution-inl.h` does not mirror the convolution TF32 guard. Patch exists in `.investigations/n23_tf32_patch.patch`. |

---

## Blackwell / CUDA Correctness Backlog

These are older Linux/CUDA findings from the Blackwell port. They are not the
current Apple Silicon task, but they still matter for the fork.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| B1 | Partial | oneDNN INT8 subgraphs | `test_self_attention[*]`, `test_batch_dot[*]`, and `test_self_attention_negative` showed pervasive INT8 numerical blow-up or crashes in DNNL matmul/batch-dot paths. | Scale fixes landed locally and AArch64 falls back instead of crashing; re-test real oneDNN INT8 on Linux x86 before closing. |
| B2 | Partial | Quantized Gluon | `test_quantize_gluon_with_forward` segfaulted under the DNNL subgraph backend after 18 earlier quantization tests passed. | AArch64 oneDNN quantized backend is disabled and native CPU quantization passes; Linux x86 oneDNN quantized Gluon still needs validation. |
| B3 | Open | Mixed dtype quantization | fp16 input/output is structurally absent in `quantize_v2`, `dequantize`, and DNNL quantize-v2 code. | Either add fp16 support or document AMP+quantize as unsupported and require fp32 casts. |
| B4 | Partial | QAT backward | STE for `quantize_v2` and QAT parameter `grad_req` fixes landed; DNNL subgraph backward exists only on a local branch and has scale-magnitude caveats. | Decide whether to push/PR the gated `_backward_sg_onednn_*` implementation after more validation. |
| B5 | Open | Mixed dtype matrix coverage | fp16/fp32 AMP, int8/fp32 quantize, and int8/fp16 combinations are separate paths. | Build an explicit coverage matrix; do not infer one path from another. |

---

## Performance / Nonblocking Engineering

These should not block basic correctness on Apple Silicon, but they affect the
quality of a public fork.

| ID | Status | Area | Issue |
|---|---|---|---|
| P1 | Deferred | cuBLASLt | fp32/fp16/bf16/fp64 env-gated cuBLASLt paths landed, but default-on, stride-aware, and INT8 follow-ups remain. |
| P2 | Deferred | TF32 | FP32 deconvolution misses the TF32 enable block used by convolution. |
| P3 | Deferred | Sparse/topk | `topk(k=10/100/1000)` is K-independent because MXNet sorts full rows then slices. |
| P4 | Deferred | Small ops | Softmax, LayerNorm, and small elementwise ops are slower than PyTorch due to multi-pass kernels and dispatch overhead. |
| P5 | Hardware | bf16 CPU | The tested Zen 2 host lacks AVX-512 BF16; oneDNN falls back to fp32 emulation. Validate on BF16-capable CPUs. |
| P6 | Resolved | Storage pool default | Round pool used more memory than Naive on ResNet-18; keep the default unless a workload proves otherwise. |

---

## Test Coverage And Integrations

| ID | Status | Area | Issue |
|---|---|---|---|
| T1 | Open | GPU tests | A full `test_operator_gpu.py` sweep was not completed on the current Blackwell build. |
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
| O2 | Open | CUDA arch matrix | Current Blackwell wheel is sm_120-only; useful public wheels need sm_80/sm_86/sm_89/sm_90 as well. |
| O3 | Open | CI | `smolix/mxnet` still lacks CI. Even a small build plus DNNL subset would catch regressions. |
| O4 | Open | Release | GitHub Release and wheel publication are not automated. |
| O5 | Open | Changelog | No clear release notes for CUDA 13, cuDNN 9, oneDNN v3, quantization, and Apple Silicon changes. |
| O6 | Open | README/docs | README and docs still read like archived Apache MXNet and omit modern dependency guidance. |
| O7 | Open | Packaging ecosystem | No conda/system package story for this fork. |
| O8 | Strategic | Upstream status | Apache MXNet was archived on 2023-11-17; all future fixes must live in this fork or downstream users must migrate. |
| O9 | Strategic | oneDNN cadence | Future oneDNN major releases will likely require repeated porting work. |

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

Before calling the Apple Silicon CPU port good enough for broader testing:

1. Decide whether A6 needs code changes now or only documented follow-up tests.
2. Run the full C++ and Python Apple Silicon sweeps after the T12/T13 fixes.
3. Re-run OpenCV/image/DataLoader tests in the final build profile.

Before shipping another public Linux/CUDA preview wheel:

1. Re-test B1/B2 on the newest master.
2. Run T1 full GPU operator sweep.
3. Resolve or document C1-C8.
4. Add at least minimal CI.
5. Update README, dependency docs, release notes, and wheel packaging notes.

Pointers:

- `handover.md` has the 2026-05-18 to 2026-05-19 PR/session handoff.
- `nccl_status.md`, `cudnn_autotune_v9.md`, `tf32_audit_2026-05-18.md`,
  `sparse_thrust3_bench.md`, `storage_pool_bench.md`, and
  `fp16_perf_bench.md` contain deeper historical analysis.
