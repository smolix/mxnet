# MXNet Port Issues

Canonical work tracker for the fork.  Episodic build session detail belongs
in git log, not here.

**Branch:** `cleanup/p0-p1-p2-20260522`
**Latest tag:** `v2.0.0+cu13.bw.20260522.1` on GitHub
**Local wheel:** `dist/mxnet-2.0.0+cu13.bw.20260522.2-*.whl` (USE_OPENCV=ON,
libopencv bundled at `python/mxnet/lib/`, RUNPATH `$ORIGIN/lib`)
**Validation host:** 4× RTX 4090 (sm_89), CUDA 13.0, cuDNN 9, NCCL
**macOS release tag:** `macos-arm64-slim-wheel-20260520`

Status labels:

- **Open** — known issue; no verified fix on the current branch.
- **In progress** — local work exists but is not committed and verified.
- **Partial** — first slice fixed; broader audit/coverage still pending.
- **Deferred** — cannot be verified here or out of scope for the current pass.
- **External** — owned by D2L, notebook infra, or another project.
- **Informational** — retained context, not a work item.
- **Resolved** — fix committed/verified; kept only when useful as context.

---

## Remaining Work At A Glance

This is the short queue for what still needs action. Detailed context remains in
the sections below; resolved items are retired to the appendix.

### Active queue

| Priority | Tracker | Status | Next action |
|---|---|---|---|
| P0 | B4 / XOP18 | Deferred (architectural) — NNVM/CachedOp can't reference subgraph op inputs in custom backward. | Revisit when the executor framework supports the required backward shape. |
| P0 | FS12 | **Reproduced + bisected 2026-05-23**: SIGBUS during `test_np_sum[False-int64-int64-int64-False-1-shape1]` setup (inside `MXSetIsNumpyShape` C API from `_NumpyShapeScope.__enter__`).  Crashes at ~21% through test_numpy_op.py when run as part of the file; **passes in isolation**.  Indicates prior-test corruption of the np_shape global var's page (SIGBUS on a simple atomic-flag write = munmapped/protected page).  Original report attributed it to `test_randint_generator` which was wrong — the actual crash site is much earlier in the long shard.  Apport intercepts cores so no backtrace yet. | Needs ASAN build + bisect of which earlier test corrupts the engine global state.  Or trace the test sequence under strace to catch the munmap.  Multi-day debug — defer until ASAN build is in the validation matrix. |
| P0 | D2L-Bug-2 | GPU OOM when two large models share a 24 GB GPU (CNN-Design / Sentiment-RNN OOM next to BERT-pretraining; PyTorch/JAX/TF survive same workload). | Profile storage-pool fragmentation and idle-arena reclaim; defer until profile evidence in hand. |
| P0 | D2L-Bug-3 | `natural-language-inference-bert.ipynb` dead kernel 1095s into training (no traceback). | Needs cuda-gdb + core dump; defer until host-side coredump capture available. |
| P1 | XOP12 expansion | Harness anchor lives in `test_xop12_operator_req_contract.py` (36 checks across 12 ops × 3 reqs).  Backend-parity (CPU vs GPU vs oneDNN vs cuDNN), `kWriteInplace`, hidden-output metadata, and aux-state timing dimensions still missing. | Add a backend-parity decorator and an aux-state pass; plug each new XOP fix into the harness. |
| P1 | XOP19 quantized subgraphs primary outputs | `_sg_onednn_*` (FC, conv, transformer) still bind primary output storage directly with raw DNNL memory writes; the BF16 fallback path now honors caller req (kNullOp skip + kAddTo CHECK_NE for BF16 outputs, caller req preserved for non-BF16 outputs).  Remaining: convert the primary-output write itself to CreateDNNLMem + CommitOutput. | Convert primary write; plug into XOP12 harness. |
| P1 | XOP22 contrib surfaces | Mostly cleared (`vocab`, `embedding`, `contrib.quantization`, `symbol.contrib._flatten/_regroup`, `ndarray.contrib.check_input`).  `-O` suite at 39 passed.  Remaining: ~5 invariant-style asserts in `symbol.contrib` (foreach/while_loop/cond subgraph wiring) that are deliberately invariant checks, plus deep `optimizer.py` paths if any new ones land. | Treat as ongoing; reopen if a fresh contrib repro shows a stripped check. |
| P2 | CN9 / L6 | Submodule policy items (dmlc queue, oneDNN ITT). | Track upstream; do not patch privately. |
| P2 | C4, O1, O4, O7 | Release/build matrix and Linux wheel runtime bundling. | Strategic; revisit when needed. |
| P2 | T2-T6, T11 | Ecosystem and cross-platform lifecycle coverage. | After the next major refactor. |
| External | L4, D5-D8 | D2L-owned notebook diagnostics. | Wait for fresh artifacts. |
| Remote | FP16 smoke | `tools/run_fp16_remote_smoke.sh` ready for a Zen 4+ host. | Run on target when available. |

## Immediate Linux/CUDA Execution Queue

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| L4 | External | D2L artifact intake | Wait for fresh notebook-run/output-audit artifacts; reopen only for concrete fresh runtime repros. | — |
| L6 | In progress | Compiler noise | Remaining clusters are submodule-policy items (CN9). Local source clusters cleared. | Track CN9 upstream. |
| L7 | In progress | Test scheduling | Target load envelope: 48-64 runnable tasks; cap `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=2-4`, `MKL_NUM_THREADS=1` for xdist lanes. | Keep one heavy CPU lane at a time; add GPU shards when memory is idle. |

---

## Compiler Noise Triage

These clusters came from the first Linux CUDA 13 `sm_89` build attempts on GCC
13/NVCC 13. Validation builds with `USE_OPERATOR_TUNING=OFF` completed
and linked `build/libmxnet.so`; the warning stream is in `build/mxnet-build.log`.
The goal is to reduce warning volume without papering over real CUDA or numeric
bugs.

| ID | Status | Area | Cluster | Triage action |
|---|---|---|---|---|
| CN2 | Resolved | Tuple/runtime allocation | Local overflow guards committed; `RelWithDebInfo` policy is `-Wno-error=array-bounds` + `-Wno-error=stringop-overflow` for GCC ≥ 13 with comment.  CN2 TODO checklist all `[x]`. | — |
| CN9 | Open | Third-party/link boundaries | Bundled dmlc concurrent queue still assigns `-1` into a `uint32_t` sentinel under NVCC, and oneDNN's vendored ITT assembly still lacks a non-executable-stack note. Both local patches are upstreamable, but they would dirty detached submodules rather than produce reachable MXNet commits. | Do not commit private submodule-local fixes. Track as upstream/submodule policy work or carry only through an explicit third-party patch mechanism. |

### Compiler Noise TODO Clusters

- [x] CN2 tuple/ADT allocation: local source overflow hardening rebuilt and
      passed focused runtime-container/FFI checks.
- [x] CN2 warning policy: decided — `-Wno-error=array-bounds` and
      `-Wno-error=stringop-overflow` for GCC ≥ 13 in CMakeLists.txt with
      explanatory comment.  Warning text remains visible, but `-Werror`
      doesn't trip on the flexible-tail false positives.  Commit `e2b102404`.
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
and before an Ada-specific rebuild. Current triage treats D1 as resolved for the
MXNet OpenCV wheel-dependency bug; D5/D6 are external
D2L notebook-run/output-audit ownership, D7 is external D2L/import-environment
work unless a standalone MXNet crash repro appears, and D8 is an informational
artifact-quality signal rather than an MXNet defect.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
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
post-fix GPU or DNNL core full-run summary is recorded here yet. A direct
long shard later aborted without a normal pytest summary after a hard bus error
in `tests/python/unittest/test_random.py::test_randint_generator`, but both the
focused node and the full `test_random.py` file pass in isolation on the rebuilt
binary. Treat the bus error as an order-sensitive broad-sweep blocker rather
than as a standalone random-generator or GPU driver failure.

### Cross-Operator Correctness Audit

The BatchNorm/LayerNorm investigation exposed repeated operator-contract
violations rather than one isolated implementation bug. Track these as a
clustered audit until each class has focused coverage. Trust library contracts
such as oneDNN/cuDNN semantics over MXNet wrapper assumptions when they
disagree.

Resolved XOP rows have been retired to the appendix so this table stays focused on remaining work.

| ID | Status | Area | Finding | Next action |
|---|---|---|---|---|
| XOP7 | Partial | oneDNN output req/copyback | oneDNN LayerNorm, deconv weight-grad fast path, softmax/log-softmax forward+backward, activation forward+backward, batch_norm forward, quantized batch_norm, and masked_softmax all verified via audit (sources use `CreateDNNLMem`/`CommitOutput` or explicit `kNullOp` skip + `kAddTo` CHECK_NE).  Most XOP7 "remaining" items listed earlier were stale.  XOP12 harness covers the same surface from the Python side. | Treat as effectively complete; reopen for a specific newly-discovered op without req handling. |
| XOP8 | Partial | Quantized range outputs | Audit complete: native quantized reshape, quantize, quantize_v2, oneDNN quantized activation/flatten/reshape/transpose/quantize/quantize_v2, oneDNN quantized BatchNorm, dnnl_fc range outputs, AND `_contrib_quantized_embedding` (CL-batch fix) all use `AssignQuantizedRangeOutput`.  No remaining range-output wrappers in the source that don't go through the shared helper. | — |
| XOP9 | Partial | Stochastic/resource ops | Dropout backward + MKL/cuDNN paths covered.  RNN dropout reserve-space req contract pinned in `tests/python/unittest/test_xop9_rnn_dropout_req.py` (12 cases: rnn_tanh/rnn_relu/gru/lstm × null/write/add).  Remaining: backend-specific Dropout forward `out=` covers cuDNN/MKL paths. | Cover the direct `out=` cuDNN/MKL Dropout forward path; otherwise close. |
| XOP12 | Partial | Regression-test infrastructure | Reusable parameterized req-contract harness is now in place: `tests/python/unittest/test_xop12_operator_req_contract.py` covers 12 operators × `{null, write, add}` = 36 contract checks today.  Adding a new operator is a one-line `pytest.param` row.  Backend-parity (CPU vs GPU vs oneDNN vs cuDNN), `kWriteInplace`, hidden-output metadata, and aux-state timing dimensions are still per-operator. | Plug each new XOP fix into the harness via OPS_UNDER_CONTRACT.  Add a backend-parity decorator and a hidden-output/aux-state pass as separate test files when needed. |
| XOP14 | Partial | cuDNN/library beta mapping | cuDNN activation/pooling/softmax-activation/LRN/bilinear-sampler/spatial-transformer wrappers + native LRN backward honor kNullOp/kAddTo.  Direct cuBLAS audit complete: `linalg_gemm`-going-through paths (linalg_impl.h, FC, dot, RNN, contrib/transformer.cu — 11 sites) all map `req → beta` correctly.  `quantized_fully_connected.cu` was the one remaining direct-cuBLAS site with hardcoded `beta=0`; now gated to reject kAddTo before any write (commit `a77daea17`). | If a future quantized op adds direct cuBLAS calls, audit them against the linalg_gemm pattern before merging. |
| XOP16 | Partial | Quantized inference contracts | `_contrib_quantized_elemwise_mul` dtype inference is hardened.  Quantized embedding storage contract is now pinned in `tests/python/quantization/test_xop16_quantized_embedding_storage.py` (shape + dtype + range-output value).  Embedding range outputs now go through `AssignQuantizedRangeOutput` (XOP8 fix). | Reopen for fresh storage-inference repros. |
| XOP18 | Partial | Quantized subgraph req/backward | Forward contract anchor in `tests/python/dnnl/test_xop18_quantized_subgraph_req.py` pins registration + shape contract for `_sg_onednn_selfatt_qk{,_split}` / `_sg_onednn_selfatt_valatt`.  Backward zero-grad behavior remains under B4.  Remaining: BF16 fallback per-output req preservation, primary kNullOp/kAddTo on the quantized writes (will land alongside XOP19's quantized-subgraph slice). | Convert primary/range writes in `dnnl_transformer.cc` once the XOP19 quantized-subgraph pass runs. |
| XOP19 | Partial | oneDNN descriptor/output handling | Reducer, softmax/log-softmax, batch-dot, deconv weight-grad fast path, dnnl_reshape entry, and `DNNLMaskedSoftmax` are converted or gated.  BF16 fallback paths in `_sg_onednn_conv`, `_sg_onednn_selfatt_qk{,_split}`, `_sg_onednn_selfatt_valatt` were audited: caller req preserved for non-BF16 outputs, BF16 outputs get kNullOp skip + kAddTo CHECK_NE.  Remaining: quantized oneDNN FC/conv/transformer subgraphs still bind primary output storage directly with raw DNNL memory writes. | Convert primary writes to CreateDNNLMem + CommitOutput. |
| XOP21 | Partial | Large-tensor size truncation | LayerNorm uses `index_t`; GroupNorm N-counter, ROIAlign / PSROIPool count-launch counters, BilinearSampler block count, SpatialTransformer block count, dnnl_dot bigDim products, multi_sum_sq chunks-per-tensor are all INT_MAX-guarded.  Remaining candidates: image-random kernels (rng-bound, less likely to exceed INT_MAX), NumPy indexing/linalg helpers (already use `index_t` per audit), FC inner-product (already in linalg_gemm). | Treat as effectively complete on the GPU surface; reopen for a specific INT_MAX repro. |
| XOP22 | Partial | Python validation via assert | First, second, and contrib waves landed: AMP, KVStore base / BytePS / kvstore.py, RecordIO, RTC, schedulers, Gluon data/NN constructors, Gluon Parameter, all optimizers, contrib text/vocab + text/embedding + quantization, symbol/ndarray contrib `_flatten`/`_regroup`/`check_input`.  Subprocess `python -O` suite at `39 passed` (was 19).  Remaining: ~5 invariant-style asserts in symbol.contrib subgraph wiring that are deliberately invariant checks; the converted surface is the user-input boundary. | Reopen if a fresh user-input repro shows a stripped check. |
| XOP23 | Partial | Engine/runtime hardening | Engine assert→CHECK conversions landed in `3e7ea07b3`.  Engine race stress: `Engine.WriteAfterReadChainTermination`, `Engine.RapidVarAllocDelete`, and `Engine.ShutdownRaceCreateUseDeleteCycle` in `tests/cpp/engine/threaded_engine_invariants_test.cc`.  Remaining: NCCL root/device mismatch stress (needs USE_NCCL=ON), after-fork engine cleanup (covered by OMPBehaviour.after_fork). | Cover NCCL stress once a USE_NCCL=ON build is in the validation matrix. |
| XOP26 | Partial | Plugin/output contracts | Optional plugin bugs are fixed locally: WarpCTC now returns early for `kNullOp` and rejects unsupported add requests before writing, while `plugin/opencv/opencv.py` normalizes buffers to bytes and reads `ImageListIter` files in binary mode. OpenCV plugin tests pass; WarpCTC sentinel coverage is added but skips unless the plugin is built/registered. | Re-run the WarpCTC sentinel test in a WarpCTC-enabled build. If full `kAddTo` support is required later, implement accumulation instead of the current explicit rejection. |

### Active Full-Sweep Findings

| ID | Status | Area | Finding | Next action |
|---|---|---|---|---|
| FS3 | Partial | C++ gtest | `OMPBehaviour.after_fork` now checks the child exit status and passes directly. Focused C++ filters pass: engine/lifecycle `13`, DNNL/NDArray `8`, BatchNorm `16`, narrow operator basics `10`, quiet broad filter excluding topology/perf/timing/imperative/DNNL sweeps `57/57`, `CORE_OP_RUNNER.Execute*` `6/6`, the C API symbol/CachedOp/Executor combined filter `8/8`, a combined engine/thread-local/OpenMP/runtime-container/C API/C++ executor/DNNL/BatchNorm shard at `44/44`, the previously noisy imperative shards: `IMPERATIVE.CopyOp` `1/1`, `IMPERATIVE.ActOp` `1/1`, `IMPERATIVE.CopyBackwardsOp:IMPERATIVE.ActBackwardsOp` `2/2`, storage/runtime `3/3`, and row-wise Kronecker/Khatri-Rao `14/14`. `ContextHashTest.ContextHashUnique` is a GPU-context test and fails if run with `CUDA_VISIBLE_DEVICES=`; the large imperative oneDNN layout sweep was intentionally aborted without a pass/fail summary. | Continue the remaining C++ suite in quieter shards, excluding known long topology/perf/timing cases and routing GPU-context tests through a GPU-visible command. |
| FS5 | Partial | GPU miscellaneous | Targeted non-operator GPU shards all pass on the current binary.  Latest 2026-05-23 run: batchnorm-running-stats + deconv-TF32 + device-pushpull + extensions + fork-safe-dnnl + pool-dynamic-shape + reducer-regressions = 30 passed, 2 skipped.  Earlier focused shards for AMP, fusion, profiler, NCCL, KVStore, linalg, cuBLASLt all passed.  The original full-GPU monolithic run aborted in fork-safe-dnnl (now fixed). | Treat targeted non-operator GPU shards as locally complete; reopen for a fresh GPU repro. |
| FS8 | In progress | Stale skip audit | Temporary/flaky skips are being removed only when the underlying test can be made meaningful. The repaired profiler/NCCL/KVStore stale-skip batch now passes `6 passed, 1 skipped`, with NCCL skipping only when no GPU or no NCCL feature is present. The old Gluon issue-11164 dynamic reshape/slice tests are active again in the focused repaired group, which passes `19 passed` together with the BatchNorm crash regression. Higher-order gradient and quantization GPU wrapper repairs pass together at `51 passed, 6 skipped`. | Keep stale skips under suspicion beyond this repaired batch. Do not restore broad skips; either repair the test, add a precise capability guard, or open a concrete runtime bug row. |
| FS12 | Open | Direct long shard | A direct long shard aborted around 95% completion without a normal pytest summary after a hard bus error in `tests/python/unittest/test_random.py::test_randint_generator`. The process had been visible to the GPUs; this is not evidence of a driver outage. Direct reruns now pass for `test_randint_generator` alone (`1 passed`) and for the full `tests/python/unittest/test_random.py` file (`37 passed`). | Reproduce the shard-order sequence that preceded `test_randint_generator`, preferably under gdb or with narrower predecessor chunks, then decide whether this is prior-test memory corruption, order-sensitive random state, or a host-load artifact. |
| FS13 | Partial | Broad skip/xfail debt | The lint piece is in: `tests/python/unittest/test_fs13_skip_reason_tracker_id.py` walks every `pytest.mark.skip*/xfail` and asserts the reason names a tracker, GitHub ref, or recognized capability/structural gate.  Current tree passes; future stale skips fail the lint.  Remaining work: per-skip audit deciding whether the underlying test should be re-enabled, precisely capability-gated, or kept as a tracked xfail. | Walk the current skip list and either repair, capability-gate, or document each.  Reopen as `Open` only if the lint surfaces something unexpected. |

### Focused Test Gate Before Broad Reruns

- [x] DNNL quantized conv+sum: run
      `tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_pos_conv_add3`,
      `tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_conv_bn_sum`,
      `tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_channelwise_quantize_model_skips_onednn_conv_with_sum`,
      and
      `tests/python/dnnl/test_quantization_dnnl.py::test_quantize_gluon_with_forward`.
- [x] DNNL quantization subset: rerun the Gluon quantization and oneDNN
      quantization files that exercise `_sg_onednn_conv`,
      `quantized_sg_onednn_conv`, and residual/add fusions. Combined
      `tests/python/dnnl/test_quantization_dnnl.py` plus
      `tests/python/quantization/test_quantization.py` passed `52 passed,
      2682 warnings` in 27.49s on 2026-05-21.
- [x] CPU unittest smoke: keep `tests/python/unittest/test_extensions.py` and
      the two prior Gluon GPU-memory-pressure nodes as focused checks before
      trusting another full `tests/python/unittest` summary. The extension
      smoke passed `4 passed, 1 skipped` in 1.77s on 2026-05-21 after the
      logger/resource-hygiene commits; the prior Gluon pressure nodes remain
      covered in FS5 and FS1.
- [x] GPU smoke: keep the fork-safety DataLoader, cuBLASLt FC, TF32 deconv,
      cuDNN stream/workspace, deferred-compute GPU, extension GPU, and NCCL
      metric checks ahead of any monolithic `tests/python/gpu` rerun. The
      deferred-compute plus GPU reducer shard passed `33 passed` in 5.22s on
      2026-05-21, adding a fresh direct-pytest GPU smoke after the latest
      commits.
- [x] C++ gtest: reran the BatchNorm mixed GPU/CPU filter
      `BATCH_NORM.Test2DBackwardMixed*_gpu_cpu*`; all six focused cases passed
      on 2026-05-21 before another broad `mxnet_unit_tests` pass.

---

## GitHub Delta Import

These rows were imported from the prior GitHub delta crawl so the main queue can
be processed without switching files constantly.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| GH1 | In progress | Security/tooling | Open upstream security/tooling hardening includes unsafe extraction, command injection risk in notebook conversion, insecure links, and stale docs/CD dependencies. Dependency zip extraction for the local OpenCV and libturbojpeg builders now rejects archive members that escape the intended extraction directory, with focused tests covering allowed contents and path traversal. `tools/ipynb2md.py` now invokes `jupyter nbconvert` through a subprocess argument list and cleans temporary notebooks without shelling out, with tests covering argv construction and cleanup on failure. `tools/kill-mxnet.py` now avoids local shell-built process killing, uses subprocess argument lists for local `ps`/`kill`, and quotes remote user/program arguments into a fixed shell script for SSH execution. The older-Python `build_openmp.py` tar extraction fallback now validates paths, link targets, and member types before extraction. The repo-local OpenMP, OpenCV, and libjpeg-turbo Python dependency builders now pin default archive URLs in `tools/dependencies/download_checksums.json` and verify SHA256 for both fresh downloads and cached archives, with mismatch cleanup covered by `test_dependency_build_tools.py`. The legacy shared-dependency shell downloader now quotes URL/output paths, uses curl `--fail --show-error`, removes partial outputs on curl failure, and quotes sourced dependency script paths. The OpenCV dependency downloader now has timeout/retry behavior and removes partial temp files after interrupted streams. The legacy `ci.util.download_file` helper now parses URL paths for destination filenames, uses a request timeout, rejects non-200 error bodies, preserves its 404 sentinel behavior, writes through a temp file, and keeps existing artifacts intact when replacement downloads fail. CI EC2 metadata probes are now bounded and catch `requests.RequestException`. CD S3 artifact downloads now normalize keys under the requested prefix and reject path traversal, absolute paths, and drive-letter style paths before writing. | Continue targeted hardening on broader docs/CD dependency freshness and any remaining legacy shell dependency fetchers still in scope. |
| GH2 | In progress | C/C++ inference API | C Predict and C++ inference/subgraph APIs have unresolved correctness gaps around deleted subgraph nodes and broader API parity after the first C++/C API fixes. The C++ `Executor` wrapper now keeps gradient requests aligned with the subset of inputs that have real gradient arrays, handles duplicate symbolic input names when building cached-op inputs, and avoids a recursive include through `op_suppl.h`. It also avoids recording inference `Forward(false)` calls into autograd graphs, tracks whether cached outputs came from a training forward, and reruns a training forward when `Backward()` is called after inference-only outputs. Focused `CppExecutor.*` coverage passes. The C API now has focused coverage for one shared thread-safe `CachedOp` handle invoked concurrently through `MXInvokeCachedOp` with caller-owned output handles. `MXSymbolGetInputSymbols` now uses `Symbol::ListInputs(kAll)` so bare variable symbols are returned and repeated variable inputs are deduplicated consistently with `MXSymbolListArguments`; focused `CAPISymbol.*` coverage passes. `CutGraphInputs` now deduplicates repeated boundary entries, including `NodeEntry.version`, so cut-subgraph APIs do not return duplicate external inputs when the marked subgraph consumes the same input twice. `MXSymbolGetChildren` now also deduplicates deleted-subgraph boundary children by full `NodeEntry`, with C inference API coverage for `ListArguments`, `GetInputs`, `GetChildren`, and `InferShape`. | Continue broader C/C++ symbol API parity after backend subgraph generation; no remaining concrete deleted-subgraph repro is open from the imported list. |
| GH4 | Resolved | Resource hygiene | RecordIO handles, logger file lifetime, operator-module signature generation (base.py first/second-file leak fixed), notebook output scan, im2rec, and rec2idx all use try/finally cleanup.  Sweep of `tools/` + `python/mxnet/` finds no remaining raw `open()` without protection. | — |
| GH6 | In progress | Data/datasets | Dataset/RecordIO issues remain around any fresh transform or data-pipeline repros after the focused fixes. The C++ `RecordFileDataset` handle now resets its thread-local RecordIO reader when switching between `.rec` files, `ImageFolderDataset` accepts an explicit `classes=` order for split-stable labels while rejecting invalid class lists, and image dataset handles now choose a separator absent from all image paths instead of always using `|`, so legal paths containing `|` are not split into bogus `ImageSequenceDataset` entries. `MultiBoxPrior` now has focused imperative/symbolic expected-value coverage for explicit `steps`, `offsets`, and `clip=True`. `RandomRotation(rotate_with_proba=0.0)` now preserves extra transform arguments such as labels instead of dropping them on the skip path. Subagent attempts for additional GH6 dataset-edge scans timed out before code/test output was produced. | Continue with any fresh dataset/transform repros. |
| GH7 | Deferred | Distributed training | Horovod KVStore lacks a barrier API, distinct from ps-lite/NCCL backlog. | Defer unless Horovod support is explicitly in scope. |
| GH8 | Deferred | Linux CPU performance | FlexiBLAS detection, transparent huge pages, and `parallel_for` grain tuning need Linux benchmarking. | Revisit after correctness and benchmark harnesses are in place. |
| GH9 | Deferred | TensorRT | TensorRT upgrade/build work is stale and separate from core CUDA validation. | Defer until CUDA CI exists. |

---

## Linux/CUDA Validation Backlog

These are real findings that need validation on the current Linux/CUDA host.
Use targeted repros before launching full GPU sweeps.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| C4 | Open | CUDA build matrix | Ada/Hopper/Blackwell architecture coverage and older CUDA 12.x compatibility need CI coverage. | Validate `sm_89` here; leave CUDA 12.x and dedicated Blackwell to later runners. |

---

## Blackwell / CUDA Correctness Backlog

These are older Linux/CUDA findings from the Blackwell port. They are not the
current Apple Silicon task, but they still matter for the fork.

| ID | Status | Area | Issue | Next action |
|---|---|---|---|---|
| B4 | Open | Full QAT backward | STE for `quantize_v2` and QAT parameter `grad_req` fixes landed, but full QAT/DNNL subgraph backward is not implemented on current `master`. The inference-oriented `_sg_onednn_*` subgraph ops still have zero-gradient or missing backward bodies for training-through-quantized-graph use cases, so this remains a real implementation task rather than a validation-only item. The 2026-05-19 handover reported a local-only gated implementation on branch `fix/fu6-qat-subgraph-backward` with commits `74b62dc27` and `a24610a02`, env gate `MXNET_QAT_SUBGRAPH_BACKWARD=1`, source `src/operator/subgraph/dnnl/dnnl_qat_backward.cc`, and test `tests/python/dnnl/test_fu6_qat_subgraph_backward.py`; none of those refs/files are present in this clone. The handover's `13 PASS, 4 XFAIL` note was from an earlier WIP binary, not a fresh rebuilt-library validation. The current test file now has strict xfail coverage showing that enabling the historical gate still leaves FC/Conv input gradients at zero until real `_sg_onednn_*` backward bodies exist. A narrow QAT test-infrastructure step now exercises the real `qat=True` path for FC weight-gradient xfail coverage and adds a passing `test_qat_quantized_grad_req_write` guard; the focused file reports `14 passed, 6 xfailed`. | Implement a fresh gated `_backward_sg_onednn_*` design for QAT training, at minimum covering the Dense/FC and Conv subgraph paths plus their range/scale inputs and `grad_req` semantics. Recover the old branch only as reference material if possible; validate against a fresh rebuilt Linux x86 oneDNN binary, check QAT scale-gradient magnitude/numeric sanity, and add focused regression tests before PR. |

---

## Performance / Nonblocking Engineering

These should not block basic correctness on Apple Silicon, but they affect the
quality of a public fork.

| ID | Status | Area | Issue |
|---|---|---|---|
| P1 | Deferred | cuBLASLt | fp32/fp16/bf16/fp64 env-gated cuBLASLt paths landed, but default-on, stride-aware, and INT8 follow-ups remain. |
| P3 | Deferred | Sparse/topk | `topk(k=10/100/1000)` is K-independent because MXNet sorts full rows then slices. |
| P4 | Deferred | Small ops | Softmax, LayerNorm, and small elementwise ops are slower than PyTorch due to multi-pass kernels and dispatch overhead. |
| P5 | Hardware | bf16 CPU | The tested Zen 2 host lacks AVX-512 BF16; oneDNN falls back to fp32 emulation. Validate on BF16-capable CPUs. |

---

## Test Coverage And Integrations

| ID | Status | Area | Issue |
|---|---|---|---|
| T2 | Open | Out-of-tree users | GluonNLP, Sockeye, AutoGluon, and DGL compatibility is untested. DGL is explicitly out of scope for the current Apple Silicon work. |
| T3 | Partial | Distributed training | Single-machine local/device/NCCL tests passed on Blackwell; multi-machine ps-lite rendezvous was not deployed. |
| T4 | Open | Python versions | Python 3.13+ is untested. |
| T5 | Partial | NumPy ABI | NumPy 2.x compatibility is not established; several failures already came from API drift. |
| T6 | Partial | DLPack | CPU byte-offset handling and NumPy interchange are fixed; PyTorch/JAX and CUDA interop remain stale/untested. |
| T11 | Open | Cross-platform lifecycle coverage | Apple Silicon now has focused async lifecycle tests. Mirror and validate the same patterns on Linux x86 CPU/oneDNN and CUDA before treating them as platform-complete. |

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
| O4 | Open | Release | GitHub Release and wheel publication are not automated. The public GitHub release `v2.0.0+cu13.bw.20260518.2` was published on 2026-05-18 at 20:30:11 UTC and predates later master fixes from the 2026-05-19 handover, so do not treat that wheel as current-source provenance. |
| O7 | Open | Packaging ecosystem | No conda/system package story for this fork. |
| O8 | Strategic | Upstream status | Apache MXNet was archived on 2023-11-17; all future fixes must live in this fork or downstream users must migrate. |
| O9 | Strategic | oneDNN cadence | Future oneDNN major releases will likely require repeated porting work. |
| O11 | In progress | Variant versioning | Source `python/mxnet/libinfo.py` still carries the prior CUDA local version suffix, but the primary and legacy wheel setup paths now honor `MXNET_PACKAGE_VERSION` so release/staging jobs can stamp an explicit package version without editing source. The GitHub release-wheel workflow now passes the resolved tag version through that variable, validates tag-derived and manual-dispatch package versions as PEP 440 versions, and gives manual staging builds commit-specific dev versions instead of reusing a stale tag. Focused wheel-runtime packaging tests passed `14 passed`. `tools/release_provenance.py` now validates wheel metadata, embedded native commit/features, source tree commit/dirty state, and expected CUDA/OpenCV feature flags; focused tests passed `3 passed`, and the helper validated the existing local CUDA artifact with `--allow-dirty --expect-cuda on --expect-opencv off`. Existing local artifacts still predate the workflow/versioning fix, and `auditwheel show` reported the raw old CUDA artifact as `linux_x86_64` constrained to `manylinux_2_39_x86_64`. | Rebuild artifacts with a source-specific local version in a controlled policy environment before publication, then rerun provenance validation on the fresh wheel. |
| O12 | Deferred | ONNX Runtime refresh | ONNX is out of scope for the current Linux/CUDA cleanup PR, but the current upstream baseline was reviewed for later porting. ONNX Runtime `v1.26.0` was released on 2026-05-08, while current ONNX `1.21.0` maps to IR version 13 and ai.onnx opset 26. MXNet's ONNX surface is currently export-oriented through `python/mxnet/onnx/mx2onnx`, public `mxnet.onnx.export_model` / `get_operator_support`, a legacy `mxnet.contrib.onnx` alias, tests under `tests/python/onnx`, and TensorRT-side ONNX conversion through bundled `onnx-tensorrt`. CI/dependency pins are stale (`onnx==1.8.0`, `onnxruntime==1.7.0`, `protobuf==3.14.0`). The exporter currently follows `onnx.defs.onnx_opset_version()`, which would target opset 26 with modern ONNX even though most translations are opset 12/13 plus limited later shims. | Later ONNX port should refresh ONNX/ORT/protobuf/Python dependency floors, add an explicit validated target-opset policy instead of blindly using the installed ONNX default, audit schema changes after opset 13 (`Split`, `Resize`, `TopK`, reductions, quantization, `Cast`, `Transpose`, `Reshape`, `Pad`, pooling, and RNN helpers), migrate away from the `onnx.mapping` shim where practical, repair ONNX test collection, add ORT 1.26 CPU parity/checker smoke tests, and separately review the TensorRT ONNX C++ path. |

---

## Suggested Triage Order

Next session, in priority order:

1. **D2L-Bug-2** GPU OOM when two large models share a 24 GB GPU
   (CNN-Design / Sentiment-RNN OOM next to BERT-pretraining).  Storage-pool
   fragmentation / idle-arena reclaim profiling; L-sized work.
2. **D2L-Bug-3** NLI BERT dead kernel at 1095s.  Needs cuda-gdb + coredump
   capture; XL.
3. **XOP12** reusable operator contract harness — would have prevented the
   ND-0, OpenCV, argmax-size-1, BoxNMS-add, and SetDNNLMem regressions.
4. **XOP19 remaining slices**: `DNNLMaskedSoftmax` scratch-as-output;
   quantized FC/conv/transformer subgraph primary-output binding; BF16
   fallback per-output req preservation in DNNL FC / subgraph conv /
   transformer.
5. **XOP22 second wave**: KVStore/BytePS, NDArray/Symbol contrib, Gluon
   `Parameter`.
6. **Remote FP16 smoke**: `tools/run_fp16_remote_smoke.sh` on a Zen 4+
   host when available.

Long-term deferred:

- FS12 order-sensitive `test_randint_generator` bus error.
- B4 / XOP18 QAT subgraph backward (blocked on NNVM/CachedOp executor).
- CN9 / L6 dmlc queue + oneDNN ITT executable-stack note (submodule policy).

Pointers:

- `issues.md` is the canonical tracker.  Resolved rows live in the
  appendix above; keep the active tables short and let appendix entries
  hold the verification detail and commit hashes.

---

# RESOLVED / HISTORICAL

Everything below this divider is closed work or historical record.
These rows are kept for traceability and to avoid re-doing closed audits.

## Resolved Appendix

Rows in this appendix were removed from the active work tables on 2026-05-22 so the remaining queue only shows outstanding, deferred, external, partial, or in-progress work.

| Source section | ID | Status | Area | Retired issue | Retired action |
|---|---|---|---|---|---|
| Active queue | D2L-Bug-1 | Resolved | NumPy argmax GPU size-1 axis | `np.argmax(x, axis=k)` on GPU returned `[0,1,..,N-1]` instead of zeros when the k-axis had size 1 (bit d2l SSD via the (5444,1) jaccard reduction). Root cause: `reduce_kernel_M1` in `src/operator/tensor/reduce_rtc.cc` used the kernel's outer flat-output index as the `index` referenced by `FUNC = AType(OP(...), index)`. Commit `cb2cdff7a`. | Regression: `tests/python/gpu/test_d2l_argmax_size_one_axis_regression.py` (13 cases across axes/dims/keepdims/argmin). |
| Active queue | D2L-Bug-4 | Resolved | Stale `mxnet.__version__` | `mx.__version__` reported `2.0.0+cu13.bw.20260518.1` for every wheel since 18 May because `python/mxnet/libinfo.py` hard-coded the literal while only the wheel METADATA picked up `MXNET_PACKAGE_VERSION`. `python/setup.py` now writes `mxnet/_build_info.py` from the resolved version; `libinfo.py` imports from it. Commit `667498cb8`. | Regression: `tests/python/unittest/test_d2l_packaging_regression.py::test_version_matches_wheel_metadata`. |
| Active queue | D2L-Bug-5 | Resolved | Storage-manager log noise | `Using Pooled (Naive) StorageManager for ...` printed unconditionally on first allocation in each device context, polluting d2l notebook output cells. Now gated behind `MXNET_LOG_STORAGE_INIT=1`. Commit `667498cb8`. | Regression: `tests/python/unittest/test_d2l_packaging_regression.py::test_first_allocation_does_not_print_storage_manager_banner` plus opt-in variant. |
| Active queue | BoxNMS `grad_req='add'` | Resolved | Native backward segfault | `_backward_contrib_box_nms` did not declare `FResourceRequest{kTempSpace}`, so the kAddTo branch in `BoxNMSBackward` dereferenced past the empty requested-resource vector. Added the resource request in `bounding_box.cc`. Commit `667498cb8`. | `test_box_nms_backward_add_request` now passes against the rebuilt binary. |
| Carry-over | XOP7 leftover | Resolved | DNNL deconv weight-grad fast path | `CreateDNNLWeightGrad` now handles `kNullOp` (tmp + Noop, leaves out_arr untouched); `DNNLDeconvBwd::WeightsGradMem`'s kWriteTo fast path nullptr-checks `CreateDNNLData` and falls back to the helper for views/non-default layouts. Commit `667498cb8`. | Regression: `tests/python/dnnl/test_xop7_dnnl_deconv_req.py` (null/write/add). |
| Carry-over | Wheel test harness scipy | Resolved | Clean-venv collect errors | `tools/run_wheel_full_test.sh` pip-installs `scipy` + `matplotlib` so cpu_test_random / gpu_operator shards no longer collect-error. Commit `667498cb8`. | — |
| GH delta | GH4 rec2idx cleanup | Resolved | File-handle leak | `tools/rec2idx.py` closes the index file via try/finally in both `open()/close()` and `__main__` so partial indexes don't leak the FD on disk-full or interrupted runs. Commit `667498cb8`. | — |
| CN2 | CN2 warning policy | Resolved | GCC 13 flexible-tail false positives | `RelWithDebInfo` retains `-Wno-error=array-bounds` / `-Wno-error=stringop-overflow` (GCC ≥ 13) with documenting comment in CMakeLists; the open "decide policy" item has been decided as warning-only-not-error. | — |
| Active queue | XOP12 anchor | Resolved (Partial → ongoing) | Contract harness anchor | First reusable parameterized req-contract harness now lives at `tests/python/unittest/test_xop12_operator_req_contract.py`, covering 12 ops × `{null, write, add}` = 36 checks today.  Commit `187ed6b8d` (extension on top of `e00de3542`).  Backend parity / aux-state / hidden-output dimensions remain as follow-up under the XOP12 row. | Future XOP fixes should plug into OPS_UNDER_CONTRACT. |
| Active queue | XOP19 MaskedSoftmax | Resolved (Partial → contract pinned) | oneDNN output-request semantics | `DNNLMaskedSoftmaxForward` already correctly early-returns on `kNullOp` and `CHECK_NE`-rejects `kAddTo` before any output write.  Contract pinned by `tests/python/dnnl/test_xop19_masked_softmax_req.py`.  Commit `e00de3542`.  Remaining XOP19 slices listed in active row. | — |
| Active queue | XOP22 second wave | Resolved (Partial → ongoing contrib sweep) | Gluon Parameter + KVStore asserts | `gluon/parameter.py` load_dict device-mismatch, deferred-init shape, set_data uninitialized, and attribute-mismatch checks now raise RuntimeError/ValueError.  `kvstore/byteps.py` and `kvstore/kvstore.py` save/load no-updater guards raise.  Commit `e00de3542`.  `python -O` subprocess suite at 35 passed (was 19).  Contrib wrappers still pending under active row. | — |
| XOP rows | CL-3 norm channel-size guards | Resolved (XOP21 slice) | Truncation hardening | GroupNorm `N` and ROIAlign/PSROIPool `count` int-launch counters guarded against INT_MAX before narrowing.  Smoke tests in `test_xop21_large_shape_validation.py`. | — |
| XOP rows | CL-4 RNN dropout req | Resolved (XOP9 slice) | RNN reserve-space + req | RNN backward sentinel/overwrite/accumulate contract pinned across rnn_tanh/rnn_relu/gru/lstm × {null, write, add} = 12 cases in `test_xop9_rnn_dropout_req.py`. | — |
| XOP rows | CL-5 selfatt subgraph forward contract | Resolved (XOP18 anchor slice) | Quantized subgraph forward | Registration + shape contract for `_sg_onednn_selfatt_qk{,_split}` / `_valatt` pinned in `test_xop18_quantized_subgraph_req.py`. | — |
| XOP rows | CL-6 engine race stress | Resolved (XOP23 slice) | Engine invariants | `Engine.ShutdownRaceCreateUseDeleteCycle` covers create/use/delete-var rotation under the pooled engine across 8 threads × 256 cycles. | — |
| XOP rows | CL-7 base.py module-file leak | Resolved (GH4 final slice) | Resource hygiene | Operator-module signature generator wraps the second `get_module_file()` open inside the try/finally so a failure on it doesn't leak the first handle. | — |
| FS3 | FS3 image_random_crop kAddTo guard | Resolved (1 new bug from sweep) | Op req-contract | Broad gtest sweep (102/103 pass) surfaced `ImageRandomCrop.ResizePathAddToRejectedBeforeOutputWrite` failing because kAddTo was only rejected on the resize path.  Made the rejection unconditional.  Commit `f5e1aa7dd`. | — |
| Immediate Linux/CUDA Execution Queue | L0 | Resolved | Build setup | Host toolchain/runtime packages are installed, submodules are initialized, and CUDA 13 `sm_89` validation builds with `USE_OPERATOR_TUNING=OFF` link `build/libmxnet.so` without compiling `operator_tune.cc`. `.venv-mxnet` imports the local library and reports CUDA, cuDNN, NCCL, oneDNN, and 4 visible GPUs. | Use this validation build for focused Linux/Ada tests; keep intentionally enabled operator tuning as a separate release-build policy choice. |
| Immediate Linux/CUDA Execution Queue | L1 | Resolved locally | Apple fixes on x86 | Apple Silicon fixes for lifecycle, DataLoader, DLPack, quantization fallbacks, oneDNN test harnesses, and NumPy drift have now been validated on Linux x86 with oneDNN enabled. DataLoader fork-safety, BF16 skip policy, DNNL batch-dot coverage, OpenMP fork handling, extension optional-artifact behavior, DNNL adaptive-pooling triage, broad CPU unittest at `1752 passed, 91 skipped`, focused C++ lifecycle/DNNL/BatchNorm/operator filters, controlled CPU operator lane at `1092 passed, 1 skipped`, several focused GPU harness checks, DNNL core/subgraph/quantization serial reruns, the DNNL quantized conv+sum regression, the NumPy-heavy GPU operator shard at `3131 passed, 1 skipped`, and the classic GPU operator complement now pass locally after the TF32 parity-test harness fix. | Keep these shards in the release validation matrix; reopen only for a fresh Linux x86 regression. |
| Immediate Linux/CUDA Execution Queue | L2 | Resolved | CUDA smoke | The d2l diagnostics were produced from a pre-Apple-port wheel that lacked Ada kernels. After rebuilding for `sm_89`, the standalone d2l GPU probes are OK across the 4-GPU host. | Treat the old no-kernel-image failures as stale for this host; use current D2L notebook-run/output-audit artifacts under L4 for any notebook follow-up. |
| Immediate Linux/CUDA Execution Queue | L3 | Resolved locally | CUDA regression batch | Targeted CUDA tests have cleared cuDNN/TF32 deconv, cuDNN large-channel conv/deconv probes, cuDNN frontend-autotune smoke and explicit no-plan fallback, cuDNN stream/workspace regression coverage, cuBLASLt GEMM/FC/dtype/strided checks plus a same-process 4-GPU threaded cuBLASLt stress, fp16 batch-dot parity, linalg temp storage, deferred-compute GPU, reducer regressions, NumPy einsum GPU, KVStore local/device/GPU checks, NCCL multiprocess, histogram GPU edge cases, Proposal/MultiProposal checked-arithmetic coverage, a 26-node bounded GPU operator micro-shard, the NumPy-heavy operator shard at `3131 passed, 1 skipped`, the classic operator complement at `1905 passed, 5 skipped` plus the repaired depthwise parity rerun, and the NCCL single-process bandwidth check after converting the hard bandwidth gate into a metric. Extension GPU tests now skip when optional shared libraries were not built. | Keep direct pytest/Python GPU commands in the validation matrix; leave Blackwell-specific validation under C4/O2. |
| Immediate Linux/CUDA Execution Queue | L5 | Resolved | Tracker cleanup | Duplicate issue trackers and stale markdown reports have been imported here or removed from the repo. Current D2L diagnostic logs are treated as local artifacts; only executable repro tools should be retained intentionally. `origin/docs/handover-2026-05-19` is an ancestor of `master` with no branch-only diff; the useful handover facts are already retained under B4, O4, and O11. | Keep `issues.md` as the processing queue; retain only active investigation notes and executable d2l repro tools. |
| Immediate Linux/CUDA Execution Queue | L8 | Resolved locally | Build freshness | After the latest code/test fixes, CMake was reconfigured, `build/libmxnet.so` was rebuilt, and the rebuilt library was synced into `python/mxnet/libmxnet.so`. Fresh smokes report `CUDA=True`, `OPENCV=False`, CXX11 ABI `1`, an embedded native commit matching the rebuilt source commit, and a passing GPU sum. Focused post-sync checks pass for the depthwise TF32 parity rerun, C API symbol/CachedOp/C++ executor shard `8/8`, dependency-builder/im2rec/release-provenance Python tests `22 passed`, plus the broader CPU/GPU/DNNL shards listed under FS1-FS7. The rebuild still emits known CN2 tuple false positives and the CN9 oneDNN ITT executable-stack linker note. | Reconfigure before any release artifact if new commits land after this tracker update; otherwise the current Python package library is current for the code fixes. |
| Immediate Linux/CUDA Execution Queue | L9 | Resolved | Host GPU driver state | The CUDA error 304 / invalid-context failures came from a host NVIDIA mismatch after `/usr/bin/unattended-upgrade` upgraded the NVIDIA 580 server-open userspace packages from `580.126.20` to `580.159.03` while the loaded kernel module stayed at `580.126.20`. After reboot, `nvidia-smi` reports driver/userspace `580.159.03`, all four RTX 4090s are visible, and direct GPU pytest invocations pass focused smoke/regression checks. CUDA error 304 can still be produced by environment-prefixed probes that run without `/dev/nvidia*` inside the restricted tool context; that is not the host driver state. | Treat L9 as closed for this rebooted host. Use direct pytest/Python commands for GPU validation, and only reopen if the same CUDA failure reproduces in that command shape. |
| Compiler Noise Triage | CN1 | Resolved | Build throughput | Non-tuning builds no longer compile `src/operator/operator_tune.cc`: `tuned_op::UseOMP` is inlined to the existing non-tuning `true` behavior, CMake removes the source unless both `USE_OPERATOR_TUNING` and `USE_OPENMP` are enabled, a throwaway CPU-only non-tuning build linked, the main CUDA non-tuning build linked, and both build metadata scans omit `operator_tune.cc`. | Keep deliberately enabled operator-tuning release builds unchanged; reopen only if a release profile requires tuning and the remaining compile cost is unacceptable. |
| Compiler Noise Triage | CN3 | Resolved | dmlc optional | `include/dmlc/optional.h` emitted uninitialized warnings around `optional<T>().swap(*this)`, stream extraction, nullopt assignment, and scalar specializations. | Fixed in dmlc-core commit `d610d79` by using lifetime-aware assignment/reset/swap and by assigning parsed values only after successful extraction; standalone optional smoke passed. |
| Compiler Noise Triage | CN4 | Resolved | Reductions | Half/bfloat broadcast-reduce residual initialization now initializes `val` and `residual` independently before reducer-specific setup, and boolean product/nanprod initializes unused residual storage and uses logical AND for bool reductions on CPU and RTC GPU paths. | Clean rebuild no longer shows the CN4 residual cluster; focused CPU reducer regressions passed. GPU/autograd-covered follow-up waits for L9. |
| Compiler Noise Triage | CN5 | Resolved | CUDA type guards | Unsigned comparison warnings are reduced with type-trait negative checks in bincount, delete, nan-to-num, and NumPy random normal/location-scale validation paths. | Clean rebuild no longer shows the local CN5 unsigned-comparison cluster. Forward-only `bincount`, `delete`, `np_random`, and `np_randn` checks passed; autograd-covered random/nan-to-num checks wait for L9. |
| Compiler Noise Triage | CN6 | Resolved | Sentinel conversions | Local MXNet `size_t` sentinel initialization warnings in `np_cross` and `np_matmul` are fixed by value-initializing vectors instead of filling them with `-1`. Bundled dmlc queue offset warnings are third-party boundary work, not local CN6 blockers. | Clean rebuild no longer shows local `np_cross`/`np_matmul` sentinel warnings. `test_np_matmul_error` passed; autograd-covered `np_cross`/`np_matmul` follow-up waits for L9. |
| Compiler Noise Triage | CN7 | Resolved | Half param packing | Fused optimizer parameter packing in AdamW/AdaBelief-style code now uses typed assignment instead of `memcpy` into `mshadow::half::half_t` arrays, removing the local `-Wclass-memaccess` cluster. | Focused AdamW/AdaBelief tests passed; keep fused optimizer coverage in the CPU/GPU smoke set. |
| Compiler Noise Triage | CN8 | Resolved | Local cleanup | Cheap and low-risk local warnings are reduced: CUDA resize/transformer unused variables are removed, KVStore NCCL hidden-overload/buffer noise is cleaned up, pointwise fusion initializes the crossing-subgraph output slot, CTC wraps only the vendored moderngpu include for deprecated declarations, mshadow packet allocation initializes the compiler-visible pointer path, einsum paths initialize selected costs and pointer arrays, half CPU max-pooling uses `NegInfValue`, and runtime-handle cleanup is validated. | Post-cleanup rebuild completed; warning scan no longer shows the mshadow packet, CTC/moderngpu, einsum, or half `numeric_limits::infinity()` pooling clusters. Focused CTC/einsum/pooling/reducer/TF32 tests passed. |
| D2L Diagnostics Import | D1 | Resolved | Runtime deps | The prior MXNet wheel linked against system OpenCV 4.6 runtime libraries but did not include dependency metadata or bundled libraries, causing import-time `libopencv_imgcodecs.so.406` failures on clean hosts. Python wheel metadata cannot reliably express those system OpenCV SONAMEs, so the intended Linux/CUDA wheel path is OpenCV-off by default. The primary packaging metadata/runtime-bundling guard is committed, the Linux release-wheel workflow configures `-DUSE_OPENCV=OFF`, omits `libopencv-dev`, disables OpenCV dependency metadata with `MXNET_SETUP_ENABLE_OPENCV_DEPS=0`, and stages `libmxnet.so` at the package path that `libinfo.find_lib_path()` searches. The legacy `tools/pip/setup.py` CD path also declares `opencv-python` metadata when OpenCV is enabled and bundles resolved `libopencv_*` libraries unless `MXNET_SETUP_ALLOW_SYSTEM_OPENCV=1` is set. Focused packaging tests passed `8 passed`. A rebuilt CUDA wheel artifact was audited: metadata has no `Requires-Dist: opencv-python`, extracted `mxnet/libmxnet.so` has no `libopencv_*` DT_NEEDED entry, RUNPATH points at `$ORIGIN/lib`, pip-installed cuDNN/NCCL locations, and system CUDA paths, and a wheel-payload import reports commit `12cd3a720200a765ac0e859f0781555f64125a7a`, `CUDA=True`, `OPENCV=False`, 4 GPUs, and a passing GPU sum. | Keep D1 closed for the OpenCV dependency bug. If shipping OpenCV enabled later, bundle every `libopencv_*` dependency or explicitly document required OS packages. Remaining raw `linux_x86_64` tag/version/provenance work is tracked under O11, not D1. |
| D2L Diagnostics Import | D2 | Resolved | CUDA arch coverage | The dominant d2l failure was `cudaErrorNoKernelImageForDevice` on RTX 4090 because the tested wheel did not include `sm_89` kernels. The local `sm_89` rebuild clears the standalone GPU probe gate on this Ada host. | Keep release-wheel architecture coverage open under O2/C4; no local Ada runtime action remains for the old wheel failure. |
| D2L Diagnostics Import | D3 | Resolved | GPU scalar host sync | Six d2l RNN/optimization notebooks reported `MXNetError: could not execute a primitive` when converting GPU scalar results to host/Python. The rebuilt standalone `gpu-scalar-to-host` and `gpu-gru-scalar-to-host` probes are OK. | If current D2L notebook-run/output-audit artifacts still show full-notebook failures, require a fresh MXNet runtime repro before reopening this standalone scalar path. |
| D2L Diagnostics Import | D4 | Resolved | Transformer native crash | The crash narrowed to oneDNN `batch_dot` using packed primitive descriptors to wrap MXNet default-layout buffers. Reordering inputs into the primitive-selected descriptors fixes the attention-shaped repro; `transformer-decoder-standalone` now passes with oneDNN enabled. The matching `_sg_onednn_batch_dot` path also needed a temp-space request for those reorders, and the existing subgraph batch-dot matrix now passes. | Keep `tests/python/dnnl/test_batch_dot_attention_regression.py` and `tests/python/dnnl/subgraphs/test_matmul_subgraph.py::test_batch_dot` in the oneDNN subset; reopen transformer work only if current D2L artifacts include a fresh MXNet runtime repro beyond the fixed standalone path. |
| Current Full-Sweep Findings | FS1 | Resolved | CPU unittest | The post-rebuild lower-concurrency broad CPU unittest lane passed with `1752 passed, 91 skipped, 65300 warnings` in `977.18s`, using `-n 4` and excluding the monolithic operator files. A controlled CPU `tests/python/unittest/test_operator.py` lane then passed `1092 passed, 1 skipped` in `215.69s` with four xdist workers and BLAS/OpenMP thread caps set inside Python. Earlier failures were either optional extension libraries now skipped cleanly, GPU OOM pressure that passed in isolation, or local oversubscription from uncapped BLAS threads. | Keep these lower-concurrency CPU shards in the release validation matrix; avoid uncapped BLAS/OpenMP fan-out when xdist is active. |
| Current Full-Sweep Findings | FS2 | Resolved | oneDNN Python | Post-rebuild serial DNNL reruns now pass: `tests/python/dnnl/test_dnnl.py` passed `30`, DNNL subgraphs passed `935` with `16 skipped` and `4 xfailed`, `tests/python/dnnl/test_quantization_dnnl.py` passed `26`, and the generic quantization shard passed `26`. The earlier adaptive-pooling timeout was excess fallback work, and the reduced adaptive-pooling matrix plus DNNL conv+sum quantization checks pass. | Keep these DNNL subsets in the release validation matrix; broader INT8/self-attention and quantized-Gluon matrix work remains tracked under B1/B2. |
| Current Full-Sweep Findings | FS4 | Resolved | NCCL/multi-GPU | The old hard NCCL bandwidth threshold was load-sensitive and failed while other jobs were resident. It is now reported as a metric instead of a correctness assertion; the focused `test_nccl_bandwidth[1]` rerun passed. | Keep bandwidth values in logs for performance triage; do not fail correctness on this threshold. |
| Current Full-Sweep Findings | FS6 | Resolved locally | GPU operator | The full `tests/python/gpu/test_operator_gpu.py` file has now been covered through shards on the rebuilt Linux/Ada CUDA library. Earlier focused shards passed for convolution/deconvolution smoke, histogram, Proposal/MultiProposal checked arithmetic, subgraphs, local conv/deconv/pooling variants, mixed local operators, NumPy linalg/reduction/broadcast/elementwise/matrix-product families, optimizer/update nodes, sparse/create-sparse/SPMM, random/Gluon/optimizer paths, and the NumPy-heavy complement at `3131 passed, 1 skipped`. The classic non-NumPy complement finished at `1905 passed, 5 skipped, 1 failed`; the only failure was `test_depthwise_convolution` under TF32, where cuDNN can pick different TF32 engines for grouped and per-channel equivalent graphs. The test now disables tensor cores only inside that strict algebraic parity check, and the original failing seed rerun passes. | Treat the sharded operator file as locally complete; keep the shard set rather than a single monolithic run, and open fresh rows for any new operator repros. |
| Current Full-Sweep Findings | FS7 | Resolved | DNNL quantized conv+sum | Local mitigation disables residual conv+sum fusion during quantization, avoids channel-wise quantizing `_sg_onednn_conv` nodes that still carry `with_sum`, and fixes sum-input type restoration in DNNL conv inference. Focused rebuilt-library checks passed: `test_pos_conv_add3`, `test_conv_bn_sum`, `test_channelwise_quantize_model_skips_onednn_conv_with_sum`, and `test_quantize_gluon_with_forward`. | Keep these in the DNNL quantization subset for broad reruns. |
| Current Full-Sweep Findings | FS9 | Resolved locally | Gluon BatchNorm crash | Re-enabling `tests/python/unittest/test_gluon.py::test_batchnorm_16c` exposed heap corruption in local-statistics training BatchNorm after oneDNN convolution layouts. The fix keeps oneDNN enabled for inference/global-statistics BatchNorm paths but routes local-statistics training forward and backward through the native CPU implementation, avoiding the unsafe oneDNN backward primitive and avoiding a mixed oneDNN-forward/native-backward running-stat update. The DNNL backward cache now signs the actual reordered memory descriptors, and `fix_gamma` preserves `kAddTo` semantics. | Rebuilt `libmxnet.so`, synced `python/mxnet/libmxnet.so`, and passed BatchNorm focused checks `4 passed`: `test_batchnorm_16c_one_by_one_training_backward`, `test_batchnorm_16c`, `test_batchnorm_training`, and DNNL `test_batchnorm`. The broader repaired Gluon stale-skip group passed `19 passed`. |
| Current Full-Sweep Findings | FS10 | Resolved locally | Higher-order gradients | The failures came from test-source misuse of `autograd.grad(..., variables=x)`, which returns an `NDArray` directly; indexing `[0]` scalarized the gradient. The test now passes variable lists where list results are expected, preserving real higher-order-gradient coverage rather than disabling the file. | `tests/python/unittest/test_higher_order_grad.py` passes in isolation at `31 passed`; combined with the quantization GPU wrapper validation it passed `51 passed, 6 skipped`. |
| Current Full-Sweep Findings | FS11 | Resolved locally | Quantization GPU wrapper | The CUDA 304 quantization report did not reproduce as a driver/runtime failure. The wrapper now imports the CPU quantization tests from the correct package path, provides the private legacy-NDArray-semantics fixture, sets/restores the default GPU device through a module fixture instead of at import time, and precisely skips CPU-only or unsupported GPU quantized paths such as bfloat16 scalar copy, quantized transpose/reshape, quantize-model, and RNN quantization. | `tests/python/test_quantization_gpu.py` passed `20 passed, 6 skipped`; combined with higher-order-gradient coverage it passed `51 passed, 6 skipped`. |
| GitHub Delta Import | GH3 | Resolved | Autograd/Gluon semantics | The imported autograd/export repros now have focused fixes or coverage. `autograd.grad(heads, variable)` preserves the documented single-NDArray return shape, wraps gradients to match each variable's array class, and no longer duplicates gradients for explicitly requested attached intermediates in the native imperative engine. Legacy and NumPy `attach_grad(grad_req='add')` accumulate into existing buffers while preserving dtype and array class, non-leaf `attach_grad()` preserves upstream gradient flow, NumPy float16 gradient dtype is covered, custom `autograd.Function` backward handling accumulates integer add requests and skips null-request inputs correctly, custom backward callbacks wrap gradient handles using the actual forward array classes, Gluon export/import preserves float64 Dense parameter dtype and output parity, and mixed NumPy binary backward coverage checks fp16/fp32 promotion and input-gradient dtype preservation. Focused Python and rebuilt-native checks pass, including the attached-intermediate `autograd.grad()` regression. | Keep the focused CPU/CUDA parity checks in validation; reopen only for a fresh autograd/export repro. |
| GitHub Delta Import | GH5 | Resolved locally | Operator correctness | The imported stale operator PRs now have focused fixes or coverage for the listed items; no fresh GH5-specific repro was found in the latest CPU, DNNL, sparse, shape-inference, small GPU, or sharded `test_operator_gpu.py` sweeps. CPU `mx.nd.argsort` accepts float16 inputs through the ordering dispatch path, `mx.nd.linspace(..., dtype='int32')` matches NumPy integer flooring semantics, RReLU training forward samples into the mask buffer, NumPy min/max reductions cover infinity cases, `ModulatedDeformableConvolution` uses `npx.slice` on the layout channel axis, and Gluon `GroupNorm` ignores disabled affine parameters. Additional validation passed for adaptive/max pooling, take/gather/scatter, batch-dot attention and large concat, NumPy broadcast/repeat/take/where, deconv TF32, dynamic pooling, fp16 batch-dot, embedding backward NaN, reducer/deferred-compute, grad_req/add/inplace dtype, sparse where/reshape/broadcast, partial shape inference, and DNNL adaptive-pooling/softmax/concat/elemwise-add. | Treat GH5 as closed for the stale imported bundle; open fresh operator rows for new concrete failures rather than reopening the umbrella. |
| Active Apple Silicon / CPU Queue | A6 | Resolved locally | Resource shutdown | Custom-op workers and thread-local temp resources needed shutdown-order hardening. | Mirror the new lifecycle tests on Linux x86/CUDA before calling this platform-complete. |
| Active Apple Silicon / CPU Queue | A7 | Resolved locally | macOS multiprocessing | Some Apple Silicon/macOS environments allow Python `multiprocessing.shared_memory` probes to pass while MXNet `cpu_shared` allocation fails with `shm_open: Operation not permitted`, breaking multiworker `DataLoader`. | Validate the pickle-transport fallback on Linux x86/CUDA; it should remain inactive there when `cpu_shared` works. |
| Active Apple Silicon / CPU Queue | A8 | Resolved locally | macOS wheel | Build a slim optimized Apple Silicon wheel with Accelerate, oneDNN, OpenMP, OpenCV, and libjpeg-turbo, but without ONNX/MPS/GPU. | Install and smoke-test the wheel on another Apple Silicon machine if available; then move to Linux/CUDA validation. |
| Linux/CUDA Validation Backlog | C1 | Resolved | Histogram CUDA | CPU histogram validation fixes now have CUDA parity coverage: existing GPU histogram and NumPy histogram nodes pass, and `test_histogram_gpu_edge_and_invalid_bins` covers the right-edge and invalid-bin cases directly on GPU. | Keep the focused histogram GPU nodes in the CUDA smoke shard. |
| Linux/CUDA Validation Backlog | C2 | Resolved | Proposal CUDA | The CUDA `Proposal` and `MultiProposal` paths now use checked arithmetic for shape products, NMS mask sizing, allocation byte counts, host-vector sizes, pointer offsets, and kernel `int` launch counts. The rebuilt GPU `test_multi_proposal_op`, the CPU `test_multi_proposal_op`, and the histogram-or-MultiProposal GPU shard pass. | Keep `test_multi_proposal_op` and the histogram/proposal shard in CUDA smoke coverage; no active Proposal-specific overflow/narrowing work remains. |
| Linux/CUDA Validation Backlog | C3 | Resolved | cuBLASLt | Direct cuBLASLt GEMM, FC, dtype, and strided tests pass together: `27 passed, 7 skipped`. A same-process threaded stress using 2 threads per GPU across all 4 RTX 4090s passed dot and strided batch-dot loops with mixed fp32/fp16 work. A prior GPU1 environment-prefixed rerun failed with CUDA 304 in subprocess children, matching the known restricted command-context artifact and not a runtime regression. | Keep the direct cuBLASLt subset and same-process threaded stress pattern in CUDA smoke/performance triage; keep CUDA 304 environment-prefix failures out of correctness accounting. |
| Linux/CUDA Validation Backlog | C5 | Resolved | cuDNN frontend | Representative cuDNN convolution/deconvolution smoke now passes on Ada, the higher-risk large-channel probes `test_convolution_large_c` and `test_deconvolution_large_c` pass, frontend plan selection works on a large-C NCW convolution including a zero-workspace variant, and explicit no-plan fallback is now covered by `MXNET_CUDNN_FORCE_NO_HEURISTIC_PLANS=1` in `tests/python/gpu/test_cudnn_frontend_no_plan_fallback.py`. The fallback builder also skips unsupported engine-config finalization statuses instead of aborting before trying later fallback engines. | Keep frontend-autotune and no-plan fallback tests in CUDA coverage; reopen only for a fresh cuDNN frontend/fallback runtime failure. |
| Linux/CUDA Validation Backlog | C6 | Resolved | cuDNN streams | The old skipped multi-stream regression compared cuDNN against a weak cross-backend training path. It now runs cuDNN directly under `MXNET_GPU_WORKER_NSTREAMS=1/2` and `NaiveEngine`, `ThreadedEngine`, and `ThreadedEnginePerDevice`, using deterministic inputs and output gradients with a NumPy oracle for forward, data-gradient, weight-gradient, and bias-gradient results. The focused node passes locally with the adjacent conv/deconv guard. | Keep `test_convolution_multiple_streams` in CUDA smoke coverage; if future failures appear, treat them as cuDNN wrapper behavior against the independent oracle rather than as non-cuDNN numeric drift. |
| Linux/CUDA Validation Backlog | C7 | Resolved | CUDA kernels | The targeted CUDA edge shard now passes split, reshape, reducer, kernel-error, zero-size tensor, concat, and `zero_sized_dim` coverage through the 26-node bounded GPU operator micro-shard plus `tests/python/gpu/test_reducer_regressions.py`. | Keep the micro-shard in CUDA smoke coverage; broader operator completeness remains tracked under FS6/T1. |
| Linux/CUDA Validation Backlog | C8 | Resolved | TF32 deconvolution | Tracker was stale: `cudnn_deconvolution-inl.h` now mirrors the convolution TF32 guard and `tests/python/gpu/test_deconv_tf32.py` exists. The Ada rerun passed `4 passed`. | Keep the focused test in CUDA smoke coverage; no current TF32 deconvolution fix is needed. |
| Blackwell / CUDA Correctness Backlog | B1 | Resolved | oneDNN INT8 subgraphs | The current Linux x86 oneDNN build now passes the full INT8 matmul/self-attention subgraph file: `tests/python/dnnl/subgraphs/test_matmul_subgraph.py` reported `64 passed`. The earlier `test_self_attention[*]`, `test_batch_dot[*]`, and `test_self_attention_negative` blow-ups no longer reproduce on this rebuilt library. | Keep the matmul subgraph file in the oneDNN validation subset; reopen only for a fresh INT8 subgraph runtime or accuracy failure. |
| Blackwell / CUDA Correctness Backlog | B2 | Resolved | Quantized Gluon | The historical `test_quantize_gluon_with_forward` segfault no longer reproduces, and the full DNNL quantization file now passes on Linux x86: `tests/python/dnnl/test_quantization_dnnl.py` reported `26 passed`. The related conv+residual-sum quantization checks under FS7 also pass. | Keep `test_quantization_dnnl.py` and the FS7 focused conv+sum checks in oneDNN validation; reopen only for a fresh quantized-Gluon or DNNL quantization failure. |
| Blackwell / CUDA Correctness Backlog | B3 | Resolved | Mixed dtype quantization | True fp16 quantize/dequantize kernels remain unsupported by design, but AMP now treats `_contrib_quantize_v2` and `_contrib_dequantize` as FP32 boundary ops instead of dtype-neutral fp16/fp32 ops. Focused tests cover the AMP cast boundary, dequantize output staying fp32, direct float16 `quantize_v2` rejection, and AMP coverage now includes `_contrib_quantized_npi_add`. | Documented behavior is fp32 at AMP quantization boundaries; reopen only if native fp16 quantization support is explicitly required later. |
| Blackwell / CUDA Correctness Backlog | B5 | Resolved | Mixed dtype matrix coverage | A focused mixed-dtype matrix now covers the distinct fp16/fp32 AMP, int8/fp32 quantize/dequantize, and int8/fp16 boundary cases rather than inferring one path from another. The new `tests/python/unittest/test_mixed_dtype_matrix.py` passed `5 passed`. | Keep the matrix in the unit validation set; reopen only for a fresh dtype-combination gap. |
| Performance / Nonblocking Engineering | P2 | Resolved | TF32 | Duplicate of C8; keep the actionable CUDA validation under C8 and do not track a second TF32 deconvolution row here. |  |
| Performance / Nonblocking Engineering | P6 | Resolved | Storage pool default | Round pool used more memory than Naive on ResNet-18; keep the default unless a workload proves otherwise. |  |
| Test Coverage And Integrations | T1 | Resolved locally | GPU tests | `tests/python/gpu/test_operator_gpu.py` is now covered on the current Linux/Ada host by systematic shards rather than one monolithic process. The host driver mismatch is resolved, focused GPU smoke/regression tests pass, the broad NumPy-heavy complement passed `3131 passed, 1 skipped`, the classic complement passed after the TF32 depthwise parity test-harness fix, and the supporting focused shards are recorded under FS6. |  |
| Test Coverage And Integrations | T7 | Resolved | Data/image tests | Earlier `test_gluon_data.py`, `test_contrib_gluon_data_vision.py`, and `test_image.py` crashes no longer reproduce in isolation after data/batchify fixes. |  |
| Test Coverage And Integrations | T8 | Resolved | ONNX opset 18 reductions | Exporter now emits axes as input tensors for opset >=18. Broad suite had one unrelated fp16 softmax numerical failure. |  |
| Test Coverage And Integrations | T9 | Resolved | Gluon model zoo | 34/34 model zoo checks passed, aside from a pre-existing `test_parallel_download` skip. |  |
| Test Coverage And Integrations | T10 | Resolved | Custom C++ operators | 9/9 in-tree extension/custom-op checks passed on Blackwell. |  |
| Test Coverage And Integrations | T12 | Resolved locally | C++ oneDNN pooling | Full `mxnet_unit_tests` on Apple Silicon reached `IMPERATIVE.PoolingOp` failure: `outputs.size() == GetNumOutputs(param) (1 vs. 2)`. | Fixed by deriving fixture arity from parsed pooling params; focused Apple Silicon run passed. Validate in Linux x86 oneDNN CI. |
| Test Coverage And Integrations | T13 | Resolved locally | C++ oneDNN convolution | Full `mxnet_unit_tests` on Apple Silicon reached `IMPERATIVE.ConvOp` oneDNN-vs-native data mismatches in `tests/cpp/include/test_dnnl.h:637`. | Fixed as a test-harness comparison issue by replacing raw `memcmp` with tolerant numeric comparison; focused Apple Silicon run passed. Validate in Linux x86 oneDNN CI. |
| Test Coverage And Integrations | T14 | Resolved locally | Apple Silicon oneDNN fallbacks | Fresh-process Python model and operator tests still found AArch64 Xbyak crashes outside the earlier quantized/RNN scope. | Added fallback coverage for the remaining failing float primitive families and a fresh-process regression test. Validate on Linux x86 oneDNN CI to ensure those paths remain enabled there. |
| Test Coverage And Integrations | T15 | Resolved locally | Optimized Apple Silicon C++ sweep | Optimized `-O3 -DNDEBUG -g0 -mcpu=apple-m1` C++ test build completed with OpenMP, OpenCV, oneDNN, Accelerate, and libjpeg-turbo enabled. | `mxnet_unit_tests` passed 89/89 in `build-macos-arm64-slim-optimized`. |
| Test Coverage And Integrations | T16 | Resolved locally | Optimized Apple Silicon Python sweep | Full optimized Python unittest sweep was run before the final DataLoader fallback. It finished with exactly one failure, the known macOS `cpu_shared` DataLoader failure that was fixed afterward. | Pre-fix result: `1 failed, 14045 passed, 67 skipped, 65347 warnings in 1:26:34`; post-fix targeted DataLoader checks passed. Full re-run was skipped because the final fix is Python-only and targeted coverage passed. |
| Test Coverage And Integrations | T17 | Resolved locally | macOS wheel smoke | Final ONNX-free macOS arm64 wheel was installed into a fresh UV-created venv. | Import succeeded; `mx.__version__ == 2.0.0+macos.arm64.20260520`; ONNX specs were absent; `OPENMP`, `OPENCV`, and `ONEDNN` reported enabled; `mx.nd.ones((2,3)).sum()` returned `6.0`. |
| Build, Release, And Operations | O2 | Resolved locally | CUDA arch matrix | The old public Blackwell wheel was effectively sm_120-only; useful public wheels need sm_80/sm_86/sm_89/sm_90/sm_120 coverage. `BUILDING.md` now documents the release recipe with explicit `sm_80/sm_86/sm_89/sm_90/sm_120` SASS plus `compute_120` PTX, leaves `CMAKE_CUDA_ARCHITECTURES` unset, and `tests/python/unittest/test_cuda_arch_policy.py` statically guards the source CMake matrix and `CMAKE_CUDA_ARCHITECTURES=OFF` behavior. The policy test passed `2 passed`. |  |
| Build, Release, And Operations | O3 | Resolved locally | CI | A lightweight GitHub Actions workflow now runs preflight checks, Python compile/import checks, focused unit tests, and bounded CMake/source sanity checks so `smolix/mxnet` has a basic regression gate. YAML parsing, extracted shell syntax, the local run block, and cached diff checks passed before commit. |  |
| Build, Release, And Operations | O5 | Resolved locally | Changelog | `CHANGELOG.md` now records release notes for the CUDA 13/cuDNN 9/Linux wheel line, oneDNN and quantization validation, Apple Silicon/macOS arm64, and packaging caveats. |  |
| Build, Release, And Operations | O6 | Resolved locally | README/docs | README and static-site install/build guidance now describe this maintained fork, Linux CUDA dependency expectations, the OpenCV-off CUDA wheel behavior, and the split between pip-provided cuDNN/NCCL packages and the system CUDA toolkit. |  |
| Build, Release, And Operations | O10 | Resolved locally | macOS wheel artifact | Slim optimized CPython 3.12 macOS arm64 wheel built with `-mcpu=apple-m1`, Accelerate, oneDNN, OpenMP, OpenCV, and libjpeg-turbo; ONNX and MPS/GPU are excluded. | Artifact: `dist/mxnet-2.0.0+macos.arm64.20260520-cp312-cp312-macosx_11_0_arm64.whl`; SHA256 `3953e9ad44934259ab0518f2c00f29bd0bd7bff8d959c1093fd1d3c2371a20af`. The `dist/` directory is intentionally not part of the PR. |

## Resolved Cross-Operator Audit Items

Apple Silicon CPU queue collapsed into this resolved section: there are no
active Apple Silicon-only rows; shared validation continues under
Linux/CUDA, full-sweep, and lifecycle rows in the active sections above.

| ID | Area | Resolution | Verification / disposition |
|---|---|---|---|
| XOP1 | Norm state semantics | Native CUDA BatchNorm with `cudnn_off=True` and SyncBatchNorm now update moving mean/variance in forward instead of backward. | Focused GPU regressions pass for forward-visible running stats. Keep these in the CUDA normalization shard. |
| XOP2 | Norm affine semantics | BatchNorm and SyncBatchNorm no longer mutate gamma for `fix_gamma=True`; native CUDA, cuDNN, and SyncBatchNorm fixed-gamma paths preserve `grad_req='null'/'write'/'add'` semantics. | Focused GPU fixed-gamma regressions pass for native CUDA, cuDNN, and SyncBatchNorm. Keep these in CUDA smoke coverage. |
| XOP3 | oneDNN LayerNorm | oneDNN LayerNorm now publishes MXNet-visible `std = sqrt(variance + eps)`, converts visible std back to oneDNN variance for backward, and accumulates gamma/beta grads for `grad_req='add'`. Mean/std visible outputs are committed explicitly with request-aware semantics. | DNNL-sized visible-stat and gamma/beta `grad_req='add'` regressions pass. Broader oneDNN output/copyback audit remains under XOP7/XOP19. |
| XOP4 | fp16 reductions | Generic CUDA non-last-axis LayerNorm and GroupNorm now keep fp16 forward mean, std, and normalized-output scratch in fp32 until the final cast back to visible fp16 outputs, avoiding pre-division moment overflow for large reductions. | Rebuilt the CUDA multi-arch library and passed focused GPU regressions for nonconstant large-reduction LayerNorm and GroupNorm, including mean/std/out comparisons against fp32 references. CPU generic fp16 behavior was unchanged by this CUDA-only fix; reopen only for a fresh CPU repro. |
| XOP5 | Shape inference | GroupNorm, InstanceNorm, and SyncBatchNorm now reject caller-provided bad gamma/beta shapes using checked shape assignment. | Focused GroupNorm/InstanceNorm symbolic shape-inference regression passes; SyncBatchNorm object build passes and its Python regression is pending relink. |
| XOP6 | Hidden-output metadata | InstanceNorm hidden mean/var outputs now have canonical names, and CTCLoss registers `FListOutputNames` with the correct spelling. | Hidden-head JSON/list-output regression passes for InstanceNorm and CTCLoss. |
| XOP10 | Aux-state timing | `IdentityAttachKLSparseReg` now updates moving averages in forward, uses the output request for forward assignment, and backward consumes the already-updated aux state without mutating it again. | Focused forward-only/no-double-update regression passes. |
| XOP11 | Gluon affine flags | Gluon `LayerNorm(center=False/scale=False)` now substitutes zero beta and one gamma like GroupNorm instead of passing stored disabled-affine parameters. | Disabled-affine regression passes. |
| XOP13 | General output-request semantics | The concrete request-semantics suspects found in the broad scan are now fixed or explicitly gated before mutation: image resize/random crop/crop-resize, TopK `ret_typ='mask'`, native LRN backward, `_npi_average(returned=True)`, BoxNMS forward/backward visible/temp outputs, `_npi_unique` optional outputs, `sample_unique_zipfian` mixed outputs, and empty-input NumPy `sum`/`prod`/`mean`/`any`/`all` identity writes. Unsupported resize-path `kAddTo` cases now fail before output mutation. | Focused regressions were added for multi-output request mixes, sentinel preservation, `kAddTo` accumulation, and empty-output shape preservation. Narrow object builds and focused tests passed for the repaired slices; a full relink/sweep is still required before PR readiness because broad concurrent builds exposed nvcc temporary-file races. Move future discoveries to new rows instead of reopening this umbrella. |
| XOP15 | Quantized primary-output req | `_contrib_quantized_elemwise_mul`, native `_contrib_quantize`, float-input `_contrib_quantize_v2`, and `_contrib_dequantize` now honor primary-output `kNullOp` and `kAddTo`; range outputs use shared req-aware scalar helpers. oneDNN quantize/dequantize paths skip primary writes on `kNullOp`, and quantized `quantize_v2` pass-through explicitly rejects unsupported primary `kAddTo` before writing. | Focused quantization sentinel regressions for prefilled `out=` buffers and existing quantized elemwise-mul tests pass; narrow quantization object builds pass. |
| XOP17 | Quantized metadata | Quantized RNN now lists `statecell_output` when `state_outputs=True`. | Focused quantized RNN `list_outputs()` regression passes. |
| XOP20 | Image dtype validation | `src/operator/image/resize-inl.h` had an always-true dtype guard for int32/int64 validation; image resize also preserves `kNullOp` and rejects `kAddTo` before writing. | Focused dtype regression is added to `test_numpy_gluon_data_vision.py`; it skips in this OpenCV-off build but the source rebuild passed. Re-run that test in an OpenCV-enabled build. |
| XOP24 | CUDA/NCCL unchecked status | Multi-GPU status handling is now checked for the concrete findings: `cudaMemcpyPeerAsync` uses `CUDA_CALL`; KVStore P2P setup checks `cudaDeviceCanAccessPeer`, clears the already-enabled peer-access runtime error, initializes the P2P matrix, and surfaces other failures; `KVStoreNCCL` checks NCCL reduce/broadcast/group/init calls and `cudaStreamCreate`. | Healthy-path P2P coverage passes on GPU. NCCL repeated-setup coverage is added but skips in this `USE_NCCL=OFF` build; add fault-injection tests later if a CUDA/NCCL failure-injection harness becomes available. |
| XOP25 | Storage/profiler UB | `SET_GPU_PROFILER` now null-checks the profiler pointer before calling `IsProfiling()`, and Linux CPU memory info multiplies `sysinfo.freeram/totalram` by `mem_unit`. | Source rebuild passed. Optional follow-up: add sanitizer or injectable `sysinfo` tests if the test harness gains a way to mock platform memory APIs. |
| XOP27 | Visualization metadata | `plot_network()` now forms shape/type keys from the consumed output index `item[1]` and falls back to the legacy single-output key. | Multi-output visualization regression passes. |

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
| R41 | Linux wheel OpenCV guard | Runtime bundling now refuses to silently leave OpenCV SONAME dependencies to the host unless OpenCV is bundled, already staged, or explicitly allowed; setup metadata declares `opencv-python>=4,<5` when OpenCV support is enabled; the Linux release-wheel workflow explicitly builds OpenCV-off preview wheels; and the legacy `tools/pip/setup.py` CD path now applies the same OpenCV dependency/bundling policy. | `tests/python/unittest/test_wheel_runtime_packaging.py` passed `8 passed`; rebuilt CUDA wheel artifact audit confirms no `opencv-python` metadata and no `libopencv_*` DT_NEEDED entry. |
| R42 | Linux OpenMP fork test | `OMPBehaviour.after_fork` now checks the child process exit status so fork regressions cannot pass when the child fails independently. Commit `b881a9c5a`. | `build/tests/mxnet_unit_tests --gtest_filter=OMPBehaviour.after_fork` passed. |
| R43 | Optional extension artifacts | CPU/GPU extension tests now skip absent optional shared libraries instead of failing before runtime behavior can be tested. Commit `ac708930d`. | CPU extension file passed `4 passed, 1 skipped`; GPU extension file reported `2 skipped` because optional GPU/external artifacts are not built. |
| R44 | DNNL adaptive-pooling timeout | The expensive adaptive-pooling numeric-gradient test no longer runs the full row-sparse/default cross product; it keeps one small row-sparse case and a small default-storage matrix. Commit `119002dc9`. | `tests/python/dnnl/test_dnnl.py::test_adaptive_pooling` passed `5 passed`. |
| R45 | dmlc optional lifetime | `dmlc::optional` no longer swaps inactive raw storage or assigns failed stream extractions. dmlc-core commit `d610d79`; main submodule pointer recorded in `7816eb2e4`. | Standalone dmlc optional smoke compiled and ran; the configured dmlc CTest target currently reports no registered tests. |
| R46 | Product reducer warning cleanup | Product/nanprod reducers now initialize unused residual storage and use logical AND for bool reductions on CPU and CUDA RTC paths. Commit `7816eb2e4`. | Rebuilt `mxnet` with no pending work; `test_reducer_regressions.py` passed on CPU and GPU, and `test_np_prod` plus `test_device_pushpull` passed. |
| R47 | Compiler warning cleanup batch | Current local batch addresses CN4/CN5/CN6/CN7/CN8 warning clusters across reducer residuals, unsigned negative checks, local sentinel vectors, half parameter packing, KVStore NCCL naming, pointwise fusion initialization, and unused variables. | `build/libmxnet.so` rebuilt at 2026-05-21 07:47:21 UTC. Focused reducer, packaging, optimizer, DNNL conv/sum, and forward-only NumPy checks passed; GPU/KVStore/autograd-covered checks are deferred under L9. |
| R48 | Compiler warning cleanup follow-up | Current local batch addresses the remaining repo-owned CN8 warning clusters: mshadow packet allocation initialization, CTC/moderngpu include-boundary noise, einsum initialization, half CPU max-pooling negative infinity, and D2L tracker ownership cleanup. | `build/libmxnet.so` rebuilt at 2026-05-21 14:47:01 UTC after cleaning submodule-local experiments. Warning scan shows only CN2/CN9 boundary clusters remain. Focused CTC, CPU/GPU einsum, CPU/GPU pooling smoke, GPU reducer/TF32 deconv, reducer/packaging/optimizer tests passed. |

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
| ONNX | Opset 18 reduction API change handled; broader ONNX Runtime 1.26 / ONNX 1.21 refresh is deferred under O12. |
| cuDNN 9.22 bump | Depthwise conv performance improved substantially on Blackwell; smoke tests stayed clean. |
| cuDNN frontend autotune | Env-gated v9 frontend autotune path added; default remains conservative. |
| sm_120 SASS | Confirmed `12.0+PTX` already emitted sm_120 SASS; SASS-only rebuild can shrink artifacts later. |
| Sparse ops | CUDA 13 / Thrust 3 sparse benchmarks showed no port regression. |
| fp16 tensor cores | Large dense/conv fp16 tensor-core paths are near PyTorch parity on Blackwell. |

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

