# MXNet Port Issues

Updated: 2026-05-20
Current branch: `master`
Current head: `bfec1a677` (`origin/master`)
Apple Silicon follow-up merge: PR #28 from `followup/full-sweep-macos-wheel`
Linux validation host: 4x RTX 4090 Ada (`sm_89`), CUDA 13.0, cuDNN/NCCL
host dependencies and submodules installed; CUDA `sm_89` build configured; first
`mxnet` build was interrupted after a >25 minute `operator_tune.cc` compile.
Validation build with `USE_OPERATOR_TUNING=OFF` linked `build/libmxnet.so`;
editable `.venv-mxnet` imports the local CUDA/oneDNN/NCCL/cuDNN library and
reports all 4 GPUs. A oneDNN `batch_dot` descriptor bug found by the d2l
transformer repro is fixed locally and covered by a focused regression test.
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
| L0 | Resolved | Build setup | Host toolchain/runtime packages are installed, submodules are initialized, and a CUDA 13 `sm_89` validation build with `USE_OPERATOR_TUNING=OFF` linked `build/libmxnet.so`. `.venv-mxnet` imports the local library and reports CUDA, cuDNN, NCCL, oneDNN, and 4 visible GPUs. The full-feature build still has a >25 minute `operator_tune.cc` compile issue tracked by CN1. | Use this validation build for focused Linux/Ada tests; decide release-build operator tuning behavior separately under CN1. |
| L1 | Open | Apple fixes on x86 | Apple Silicon fixes for lifecycle, DataLoader, DLPack, quantization fallbacks, oneDNN test harnesses, and NumPy drift need validation on Linux x86 with oneDNN enabled. | Build CPU+oneDNN and run the focused lifecycle/DataLoader/DLPack/quantization/C++ oneDNN subsets before calling those fixes platform-complete. |
| L2 | Resolved | CUDA smoke | The d2l diagnostics were produced from a pre-Apple-port wheel that lacked Ada kernels. After rebuilding for `sm_89`, the standalone d2l GPU probes are OK across the 4-GPU host. | Treat the old no-kernel-image failures as stale for this host; move on to transformer/native-crash and notebook rerun gates. |
| L3 | Open | CUDA regression batch | Run targeted CUDA tests before expensive sweeps: cuDNN/TF32 deconv, cuBLASLt env-gated GEMM, fp16 batch-dot, linalg temp storage, NCCL single-process, and KVStore single-machine. | Use `CUDA_VISIBLE_DEVICES=0,1,2,3` where tests can use all GPUs; throttle notebook-like jobs to avoid OOM when another process is resident. |
| L4 | Open | D2L rerun gate | The rebuilt `sm_89` runtime clears the standalone GPU probe gate, and the transformer standalone repro now passes after the oneDNN `batch_dot` descriptor fix. The old d2l notebook failures still need to be re-clustered against the current library. | Rerun the notebook clusters whose standalone runtime dependencies now pass, starting with transformer and the prior dead-kernel notebooks; audit outputs rather than trusting stamps alone. |
| L5 | Resolved | Tracker cleanup | Duplicate issue trackers and stale markdown reports have been imported here or removed from the repo. | Keep `issues.md` as the processing queue; retain only active investigation notes and executable d2l repro tools. |
| L6 | In progress | Compiler noise | GCC 13/NVCC CUDA 13 builds emit enough warning noise to hide real failures, and one large translation unit currently dominates build latency. A successful `USE_OPERATOR_TUNING=OFF` rebuild captured the warning stream in `build/mxnet-build.log`. | Deduplicate warning counts by the compiler-noise clusters below, fix cheap/local warnings first, and suppress only third-party or proven false-positive families at clear boundaries. |

---

## Compiler Noise Triage

These clusters came from the first Linux CUDA 13 `sm_89` build attempts on GCC
13/NVCC 13. A logged validation build with `USE_OPERATOR_TUNING=OFF` completed
and linked `build/libmxnet.so`; the warning stream is in `build/mxnet-build.log`.
The goal is to reduce warning volume without papering over real CUDA or numeric
bugs.

| ID | Status | Area | Cluster | Triage action |
|---|---|---|---|---|
| CN1 | Open | Build throughput | `src/operator/operator_tune.cc` can spend more than 25 minutes in a single GCC 13 `-O3` compile. Disabling `USE_OPERATOR_TUNING` allowed the local CUDA validation build to link. | Test whether splitting the file, lowering optimization for that TU, or keeping operator tuning disabled only for local validation gives a safe compile-time win without changing release behavior unintentionally. |
| CN2 | Open | Tuple/runtime allocation | Repeated GCC uninitialized and array-bounds warnings flow through `include/mxnet/tuple.h`, `include/mxnet/ndarray.h`, `include/mxnet/tensor_blob.h`, `include/nnvm/node.h`, and `include/mxnet/runtime/memory.h` ADT object allocation. | Audit initialization and object-size invariants first because `-Warray-bounds` may indicate real layout assumptions; prefer a small source rewrite if real, otherwise isolate a narrow compiler diagnostic workaround. |
| CN3 | Open | dmlc optional | `include/dmlc/optional.h` emits uninitialized warnings around `optional<T>().swap(*this)`, stream extraction, nullopt assignment, and scalar specializations. | Check whether the helper leaves storage logically initialized; rewrite the noisy paths if straightforward. |
| CN4 | Open | Reductions | Half/bfloat broadcast-reduce kernels warn about possibly uninitialized residual/value pairs in `broadcast_reduce-inl.h`; boolean product/nanprod warns about `dst *= src`. | Audit the numeric kernels first because these touch user-visible reductions; specialize boolean accumulation with logical operations if applicable. |
| CN5 | Open | CUDA type guards | NVCC warns on unsigned comparisons against zero in dtype-generic kernels such as bincount, delete, nan-to-num, and random location/scale paths. | Replace value checks with type-trait guards or signed temporaries where behavior is intentional. |
| CN6 | Open | Sentinel conversions | CUDA/C++ warnings include `size_t` vectors initialized with `-1` in `np_cross` and `np_matmul`, plus queue offsets assigned `-1` in bundled dmlc concurrent queue code. | Fix local sentinel types; suppress bundled third-party headers only if the vendored code is otherwise untouched. |
| CN7 | Open | Half param packing | Fused optimizer parameter packing uses `memcpy` into `mshadow::half::half_t` arrays in AdamW/AdaBelief-style code, triggering `-Wclass-memaccess`. | Replace local half copies with typed copy/assignment helpers and run focused fused optimizer tests. |
| CN8 | Open | Local cleanup | Cheap warnings include unused variables in CUDA resize/transformer code, always-true runtime handle checks, KVStore hidden overloads/unused buffers, and mshadow packet allocation warnings. | Patch obvious unused/always-true cases after confirming no behavior change; leave higher-risk mshadow changes behind focused tests. |
| CN9 | Open | Third-party/link boundaries | Bundled CTC/moderngpu emits deprecated `std::binary_function` warnings, and the final link warns that `ittptmark64.S.o` lacks a non-executable-stack note. | Do not churn vendored code during runtime validation; route through system-header/linker-boundary treatment or targeted upstreamable fixes. |

---

## D2L Diagnostics Import

These items were imported from the prior d2l diagnostics reports and logs. They
were observed with MXNet `2.0.0+cu13.bw.20260517` before the Apple Silicon merge
and before an Ada-specific rebuild. They should be revalidated, not assumed
still present.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| D1 | Open | Runtime deps | The prior MXNet wheel linked against system OpenCV 4.6 runtime libraries but did not include dependency metadata or bundled libraries, causing import-time `libopencv_imgcodecs.so.406` failures on clean hosts. | Resolve the Linux wheel dependency policy: bundle OpenCV, disable OpenCV for the wheel, or declare/document the required system packages; check with `d2l-diagnostics/tools/check_runtime_deps.py` after wheel build. |
| D2 | Resolved | CUDA arch coverage | The dominant d2l failure was `cudaErrorNoKernelImageForDevice` on RTX 4090 because the tested wheel did not include `sm_89` kernels. The local `sm_89` rebuild clears the standalone GPU probe gate on this Ada host. | Keep release-wheel architecture coverage open under O2/C4; no local Ada runtime action remains for the old wheel failure. |
| D3 | Resolved | GPU scalar host sync | Six d2l RNN/optimization notebooks reported `MXNetError: could not execute a primitive` when converting GPU scalar results to host/Python. The rebuilt standalone `gpu-scalar-to-host` and `gpu-gru-scalar-to-host` probes are OK. | If the full notebooks still fail, debug notebook-specific shapes or native crashes rather than this standalone scalar path. |
| D4 | Resolved | Transformer native crash | The crash narrowed to oneDNN `batch_dot` using packed primitive descriptors to wrap MXNet default-layout buffers. Reordering inputs into the primitive-selected descriptors fixes the attention-shaped repro; `transformer-decoder-standalone` now passes with oneDNN enabled. The matching `_sg_onednn_batch_dot` path also needed a temp-space request for those reorders, and the existing subgraph batch-dot matrix now passes. | Keep `tests/python/dnnl/test_batch_dot_attention_regression.py` and `tests/python/dnnl/subgraphs/test_matmul_subgraph.py::test_batch_dot` in the oneDNN subset; use notebook reruns, not this standalone repro, for any remaining transformer failures. |
| D5 | External | Dead notebook kernels | `transformer.ipynb`, `natural-language-inference-bert.ipynb`, and `sentiment-analysis-rnn.ipynb` ended as `DeadKernelError` without useful Python traceback in the old diagnostics. The MXNet standalone transformer crash is fixed; rerunning those notebooks belongs to the D2L/notebook execution system. | Wait for current notebook-run artifacts from that system; reopen under MXNet only if a current runtime failure reproduces outside notebook infrastructure. |
| D6 | External | Notebook quality gate | `chapter_builders-guide/use-gpu.ipynb` had a passing stamp while stored outputs still contained MXNet GPU errors. This is an output-audit/notebook-runner correctness issue unless current outputs show a fresh MXNet runtime failure. | Let the notebook/output audit system validate stamps against stored outputs; use any fresh MXNet errors it finds as concrete repro inputs. |
| D7 | External | D2L import-time GPU probing | In restricted environments, `d2l.mxnet` can query GPUs at import time through default arguments such as `devices=d2l.try_all_gpus()`. | Track as a d2l-side lazy-default fix; not an MXNet runtime bug unless MXNet itself crashes outside sandbox constraints. |
| D8 | Informational | Cross-framework quality | Completed MXNet notebooks mostly had sane outputs; the main issue was missing GPU runtime coverage, not bad convergence in completed notebooks. | Use rerun coverage and output audit as the quality signal after runtime fixes. |

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
| Removed `FOLLOW_UPS.md` FU-6 | QAT subgraph backward bodies are not present on current `master`; B4 is canonical and requires a branch/PR decision. |
| Removed `FOLLOW_UPS.md` FU-8 | Its legacy A6/A7 labels refer to old engine-deadlock audit IDs, not current A6/A7 rows; lifecycle work is tracked by T11. |
| Removed `FOLLOW_UPS.md` FU-11 | Wide oneDNN stack/concat fallback is implemented and covered by `tests/python/dnnl/test_fu11_large_stack_concat.py`; d2l reruns are tracked under D3-D5/L4. |
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

---

## Linux/CUDA Validation Backlog

These are real findings that need validation on the current Linux/CUDA host.
Use targeted repros before launching full GPU sweeps.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| C1 | Open | Histogram CUDA | CPU histogram validation fixes need CUDA parity. | Run focused histogram CUDA tests and inspect CUDA kernels for the same bin-count/right-edge guards. |
| C2 | Open | Proposal CUDA | CUDA `Proposal` / `MultiProposal` may share the overflow and narrowing risks fixed on CPU. | Audit CUDA proposal code and add/port overflow regression tests. |
| C3 | Open | cuBLASLt | Shared workspace lifetime/race risk needs Linux/CUDA stress testing. | Run cuBLASLt GEMM/FC/dtype tests with `MXNET_USE_CUBLASLT=1` under multi-threaded and multi-GPU load. |
| C4 | Open | CUDA build matrix | Ada/Hopper/Blackwell architecture coverage and older CUDA 12.x compatibility need CI coverage. | Validate `sm_89` here; leave CUDA 12.x and dedicated Blackwell to later runners. |
| C5 | Open | cuDNN frontend | No-plan frontend autotune paths should fall back instead of aborting. | Force frontend autotune on representative conv/deconv shapes and verify no-plan cases degrade cleanly. |
| C6 | Open | cuDNN streams | The skipped multi-stream regression still needs CUDA validation. | Re-enable or run the skipped multi-stream test with all 4 GPUs visible after basic smoke passes. |
| C7 | Open | CUDA kernels | Zero-block launches and GPU split edge cases need targeted CUDA tests. | Run focused operator tests around empty tensors, split, reshape, and reductions on GPU. |
| C8 | Open | TF32 deconvolution | Tracker was stale: `cudnn_deconvolution-inl.h` now mirrors the convolution TF32 guard and `tests/python/gpu/test_deconv_tf32.py` exists. | Run the TF32 deconv test on this Ada host; resolve if correctness and timing pass. |

---

## Blackwell / CUDA Correctness Backlog

These are older Linux/CUDA findings from the Blackwell port. They are not the
current Apple Silicon task, but they still matter for the fork.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| B1 | Partial | oneDNN INT8 subgraphs | `test_self_attention[*]`, `test_batch_dot[*]`, and `test_self_attention_negative` showed pervasive INT8 numerical blow-up or crashes in DNNL matmul/batch-dot paths. The float `_sg_onednn_batch_dot` matrix now passes after the direct descriptor fix plus subgraph temp-space request, but this does not close the broader INT8 self-attention coverage. | Re-test real oneDNN INT8 self-attention on Linux x86 before closing; do not infer the whole row from the fixed float batch-dot matrix. |
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
| P2 | Resolved | TF32 | Duplicate of C8; keep the actionable CUDA validation under C8 and do not track a second TF32 deconvolution row here. |
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
| O4 | Open | Release | GitHub Release and wheel publication are not automated. |
| O5 | Open | Changelog | No clear release notes for CUDA 13, cuDNN 9, oneDNN v3, quantization, and Apple Silicon changes. |
| O6 | Open | README/docs | README and docs still read like archived Apache MXNet and omit modern dependency guidance. |
| O7 | Open | Packaging ecosystem | No conda/system package story for this fork. |
| O8 | Strategic | Upstream status | Apache MXNet was archived on 2023-11-17; all future fixes must live in this fork or downstream users must migrate. |
| O9 | Strategic | oneDNN cadence | Future oneDNN major releases will likely require repeated porting work. |
| O10 | Resolved locally | macOS wheel artifact | Slim optimized CPython 3.12 macOS arm64 wheel built with `-mcpu=apple-m1`, Accelerate, oneDNN, OpenMP, OpenCV, and libjpeg-turbo; ONNX and MPS/GPU are excluded. | Artifact: `dist/mxnet-2.0.0+macos.arm64.20260520-cp312-cp312-macosx_11_0_arm64.whl`; SHA256 `3953e9ad44934259ab0518f2c00f29bd0bd7bff8d959c1093fd1d3c2371a20af`. The `dist/` directory is intentionally not part of the PR. |
| O11 | Open | Variant versioning | Source `python/mxnet/libinfo.py` still carries the prior CUDA local version suffix. The macOS wheel was stamped in the staging tree only to avoid changing CUDA packaging metadata on this branch. | Decide a general versioning scheme for CPU/CUDA/macOS wheels before public release automation. |

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
3. Rerun the d2l notebook clusters now that the standalone GPU and transformer
   runtime probes are clean; audit notebook outputs rather than trusting stamps
   alone.
4. Work through C1-C8, B1-B5, and T11 using focused tests before full sweeps.

Before shipping another public Linux/CUDA preview wheel:

1. Re-test B1/B2 on the newest master.
2. Run T1 full GPU operator sweep.
3. Resolve or document C1-C8 and D1-D6.
4. Add at least minimal CI, including a GPU job once a runner is available.
5. Update README, dependency docs, release notes, and wheel/versioning policy.

Pointers:

- `nccl_status.md`, `cudnn_autotune_v9.md`, `sparse_thrust3_bench.md`,
  `storage_pool_bench.md`, and `fp16_perf_bench.md` contain deeper historical
  analysis.
