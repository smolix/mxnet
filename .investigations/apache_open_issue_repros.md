# Apache MXNet Open Issue Repros

Date: 2026-06-11

Scope:

- Open `apache/mxnet` issues labeled `Bug` or `Flaky`: 750 unique issues from the prior scan.
- Open PRs whose title/body/files indicate bug, security, failure, missing API, or test-failure work: 69 PRs from the prior scan.
- Follow-up broad scan: open issues without `Bug`/`Flaky` labels, external
  tracker/forum/JIRA reports referenced from GitHub, and open PRs whose titles
  were not obviously bug-related.
- Current repo: `/home/smola/mxnet` at `29aa33d8b`.
- Current wheel under test: `mxnet 2.0.0+cu13.bw.20260608.1` in `/home/smola/d2l-neu/.venv-mxnet`.

Policy:

- Runtime-verified bugs get an executable pytest repro in
  `tests/python/unittest/test_apache_open_issue_repros.py`.
- Repro tests assert the intended fixed behavior and are marked
  `xfail(strict=True)` while the bug exists. Run with `pytest --runxfail` to
  make current bugs fail hard.
- Source-verified or unknown items are not counted as runtime-verified until a
  snippet has been executed or a subprocess/static API check exercises the
  failing path.
- Items that cannot be credibly verified in this environment remain annotated
  as skipped or unknown with the reason.

Current counts:

- Runtime/static-verified executable bug repros: 233 total: 55 from the
  original open GitHub issue/PR scan and 178 from the similar-bug/current-code
  sweep. In the current worktree, 223 are fixed regression tests and 10 remain
  expected-failing repros: 1 original open issue plus 9 similar-pattern
  candidates still pending fixes. The #19655 repro lives in
  `tests/python/unittest/test_extensions.py` because it requires a custom
  extension backend; the #18575 control-flow repro lives in
  `tests/python/unittest/test_contrib_control_flow.py`; the other open-issue
  repros remain in `tests/python/unittest/test_apache_open_issue_repros.py`.
- Fixed in current worktree: issues #21176, #21119, #21111, #20936, #20657, #20605, #20577, #21156,
  #16427, #13945, #20391, #16402, #18300, #21146, #19423, #19458,
  #19422, #12286, #14695, #13953, #8817, #20180, #20076,
  #20046, #20044, #20037, #19860, #19852, #19785, #19753,
  #19686, #19683, #19659, #19021, #18919, #18770, #18669, #18575, #18563, #18078, #17936,
  #17698, #13193, #11774, and #8430; PRs #21217, #21044,
  #20491, #18792, #18583, and #17209; plus GPU issue #19628,
  symbol issue #19647, `optimize_for` issue #19655, and control-flow issue
  #18575. Current-code/similar fixes in this batch also cover
  KVStore updater context placement, NCCL updater context placement, dynamic-output
  `simple_bind` allocation, static-shape subgraph paramless-data binding,
  imperative/cached backward preservation of runtime NumPy scalar shapes, CPU
  transformer interleaved matmul optional-gradient requests, legacy while-loop
  autograd recording isolation, and dynamically loaded extension graph-pass
  dispatch through `Symbol.optimize_for`.
- Issue-side source/static-only candidates still pending runtime confirmation: 1 (#20376).
- PR-side source/static-only candidates still pending runtime confirmation: 2 (#20470, #20316).
- Broad-scan PR candidates not yet verified and not counted as current bugs: 18.
- Remaining expected-failing repros are all NumPy view/stride contract cases:
  stepped slicing, axis movement, and negative-stride flip/rot90 views. Parallel
  source review confirmed these require core strided-view metadata in NDArray/TBlob
  plus operator/API plumbing; Python-only wrappers would be incomplete.
- Baseline verification before fixes:
  /home/smola/d2l-neu/.venv-mxnet/bin/python -m pytest -q tests/python/unittest/test_apache_open_issue_repros.py
  reported 53 xfailed; running the same file with --runxfail reported
  53 failed, confirming the repros detected the baseline behavior.
- Latest checkpoint verification against local sources with the built wheel library:
  the original open-issue repro suite passed at the 50-fix threshold with
  50 passed, 3 xfailed, and 3 warnings in 36.56s. After the first
  similar-bug sweep tests were added, the similar-only slice passed as
  expected with 46 xfailed, 53 deselected, and 5 warnings in 200.85s.
  After the second-wave similar candidates were added and one XPASSing
  oneDNN FC branch candidate was removed as not verified, the selected
  second-wave slice passed as expected with 64 xfailed, 78 deselected,
  and 2 warnings in 300.97s. The full repro checkpoint after promoting
  the first 10 similar fixes passed with 60 passed, 82 xfailed, and 6 warnings
  in 368.88s. After #20605 was promoted and recursive validation/wrapper/view
  repros were added, the selected Halley/Sagan batch passed as expected with
  66 xfailed, 142 deselected, and 2 warnings in 422.40s; the selected
  Anscombe/Aristotle sparse/numeric batch passed as expected with 19 xfailed,
  208 deselected, and 2 warnings in 2.59s. Focused #19659 verification passed
  against `build/libmxnet.so` with 1 passed and 2 warnings in 0.26s under
  `--runxfail`. After auditing the current batch, the broad RunGraph post-op
  shape repair and silent CachedOp output reallocation were removed; focused
  verification still passed against the rebuilt library.
## Fix Progress

Active batch started 2026-06-11:

- Verified fixed against local sources: #20657, #20577, PR #21044, #21156,
  #16427, #13945, PR #17209, #20391, #16402, #18300, #19422,
  #12286, #14695, #13953, #8817, #20046, #19785, #19753,
  #17936, #17698, PR #20491, PR #18583, #20180, #20076, #20044,
  #19686, #19683, #19021, PR #21217, #20936, #20037, #8430,
  #19423, #19458, #18919, #18770, PR #18792, #18563, #18078,
  #13193, #19628, #18669, #11774, #19647, #19860, #21146,
  #19852, #21176, #21119, #21111, #20605, #19659, #19655, and #18575.
- Checkpoint: the full open-issue repro suite passed at 10 fixed bugs
  with 10 passed, 43 xfailed, 3 warnings in 42.23s; it passed again
  at 22 fixed repros with 22 passed, 31 xfailed, 3 warnings in 41.97s,
  at 36 fixed repros with 36 passed, 17 xfailed, 3 warnings in 41.38s,
  at 43 fixed repros with 43 passed, 10 xfailed, 3 warnings in 41.43s,
  and at 50 fixed repros with 50 passed, 3 xfailed, 3 warnings in 36.56s.
- Latest focused results after the 10-fix checkpoint: #19422, #12286,
  #14695, #13953, #8817, #20046, #19785, #19753, #17936, and
  #17698 passed with --runxfail; PR #20491 and PR #18583 then XPASSed
  in the full suite after the C++ header changes. After the 22-fix
  checkpoint, #20180, #20076, #20044, #19686, #19683, #19021,
  PR #21217, #20936, #20037, #8430, #19423, #19458, #18919,
  and #18770 passed focused --runxfail verification. PR #18792, #13193, #18563, #18078, #19628, #18669, and
  #11774, #19647, #19860, #21146, #19852, #21176, #21119, and #21111 also passed focused
  --runxfail verification after the 36-fix checkpoint. Issue #18575 passed focused
  --runxfail verification, then the xfail marker was removed and the promoted
  test plus the full legacy control-flow test file passed normally after
  disabling autograd recording for legacy while-loop bookkeeping copies while
  preserving per-iteration body recording. The full 32-fix checkpoint run found #19423, #19458,
  #18919, and #18770 as strict XPASS before their markers were removed;
  the 40-fix checkpoint run found #19628, #18669, and #11774 as strict
  XPASS before their markers were removed.
- The 221 fixed Apache repro tests are normal regression tests in
  tests/python/unittest/test_apache_open_issue_repros.py. The #19655 custom
  extension repro is a normal regression test in
  tests/python/unittest/test_extensions.py. The remaining 9 similar-bug sweep
  tests plus issue #19170 are strict expected-failing repros in the Apache repro
  file and must stay as tests before any corresponding fixes are attempted.
- Patched files so far: python/mxnet/libinfo.py, python/mxnet/recordio.py,
  python/mxnet/gluon/block.py, python/mxnet/gluon/parameter.py,
  python/mxnet/ndarray/ndarray.py, python/mxnet/numpy/multiarray.py,
  python/mxnet/base.py, python/mxnet/ndarray/register.py,
  python/mxnet/symbol/register.py, python/mxnet/ndarray/sparse.py,
  python/mxnet/gluon/utils.py, python/mxnet/gluon/nn/basic_layers.py,
  cpp-package/include/mxnet-cpp/symbol.h,
  cpp-package/include/mxnet-cpp/symbol.hpp, python/mxnet/autograd.py,
  python/mxnet/ndarray/contrib.py, python/mxnet/kvstore/horovod.py,
  python/mxnet/io/utils.py, python/mxnet/libinfo.py,
  python/mxnet/numpy/random.py, python/mxnet/ndarray/numpy/_op.py,
  python/mxnet/gluon/loss.py, python/mxnet/gluon/rnn/rnn_cell.py,
  python/mxnet/gluon/rnn/rnn_layer.py,
  python/mxnet/gluon/nn/activations.py, python/mxnet/symbol/symbol.py,
  src/imperative/cached_op.cc, src/imperative/cached_op.h,
  src/imperative/imperative_utils.h, src/imperative/infer_graph_attr_pass.cc,
  src/operator/control_flow.cc, tests/python/unittest/test_contrib_control_flow.py,
  src/kvstore/kvstore_local.h, src/kvstore/kvstore_nccl.h,
  src/operator/numpy/np_true_divide.cc, src/operator/subgraph/static_shape_subgraph_property.cc,
  src/operator/tensor/elemwise_binary_broadcast_op.h,
  src/operator/tensor/elemwise_binary_broadcast_op_extended.cc,
  src/c_api/c_api_symbolic.cc, python/mxnet/gluon/nn/conv_layers.py, python/mxnet/util.py,
  python/mxnet/_ctypes/cached_op.py, python/mxnet/image/image.py,
  src/operator/nn/pool*, src/operator/nn/batch_norm*,
  src/operator/nn/cudnn/cudnn_batch_norm.cu,
  src/operator/nn/dnnl/dnnl_batch_norm.cc,
  src/operator/nn/layer_norm.cc, src/operator/contrib/sync_batch_norm*,
  example/extensions/lib_subgraph/subgraph_lib.cc,
  src/operator/quantization/quantized_reshape*, tests/python/test_quantization_gpu.py,
  and tests/python/unittest/test_extensions.py.
- Next checkpoint: run the expanded full repro suite again before
  promoting another fixed batch, or sooner if shared/runtime behavior changes.
- Similar-bug sweep repros added so far: generated symbol/image validation
  gaps, mixed-device NumPy and legacy NDArray wrappers, no-affine
  BatchNorm/SyncBatchNorm graph loss, hybrid CPU RNN sequence-length caching,
  normalization/loss numeric overflow and NaN edge cases, InstanceNorm
  non-default-axis deferred shape inference, dynamic_unroll int32 valid_length,
  transformer optional-gradient req cases, CSR and row_sparse non-canonical
  metadata, and expanded NumPy view/copy contract cases. The recursive sweep added more sequence/image validation,
  cross-device wrapper, sparse canonicalization, numeric stability, and view
  contract repros. Fixed in the similar-bug batch so far: InstanceNorm
  non-default-axis deferred shape inference, 8 Gluon loss numeric-edge
  cases, nine generated validation-wrapper cases plus validation
  wrapper boundary checks and SequenceLast/SequenceReverse range checks,
  shape-only NumPy view helpers,
  cross-device wrapper rejection, LayerNorm large-finite normalization,
  non-hybrid BatchNorm/SyncBatchNorm large-finite normalization,
  dynamic_unroll int32 valid_length handling, GroupNorm large-finite
  normalization, SyncBatchNorm imperative no-affine graph preservation, hybrid CPU RNN
  runtime sequence-length masking, CosineEmbeddingLoss large-vector scaling,
  the remaining Gluon loss infinity/zero-weight numeric edges, and CPU
  transformer interleaved matmul optional-gradient handling, extension
  `optimize_for` argument synchronization, and dynamic extension graph-pass
  dispatch.
  Native sparse canonicalization and LP-pooling candidates now pass against
  build/libmxnet.so. The BatchNorm/LayerNorm focused audit fixed recent
  Python normalization regressions plus old native/DNNL/CUDA normalization
  bugs: invalid Gluon LayerNorm axis validation, disabled LayerNorm affine
  parameter differentiability, BatchNorm large-finite running-stat semantics,
  CPU/DNNL `fix_gamma=True` input-gamma mutation, CPU/generic/GPU LayerNorm
  stats-only output requests, CPU/GPU GroupNorm stats-only output requests,
  oneDNN LayerNorm fallback for unsupported stats ranks and zero-leading
  dimensions, SyncBatchNorm shared-buffer shape reinitialization, InstanceNorm
  dtype validation, SyncBatchNorm unsupported dtype rejection, and
  cuDNN/native-CUDA BatchNorm inference mean/invstd population including the
  native CUDA vector tail. Cached-hybrid no-affine BatchNorm graph cases now
  pass after recorded CachedOp calls anchor disabled beta inputs with a zero
  temporary gradient. One high-level mx.image.random_crop(..., interp=10)
  validation candidate was removed from the bug count because the documented
  mx.image helper accepts 10 as random interpolation, unlike the generated
  NDArray/Symbol image wrappers.

## Normalization Audit Classification

2026-06-12 BatchNorm/LayerNorm focused audit split findings into two buckets:

- Sloppy recent fixes found and corrected: Python `_scale_batch_norm_input` kept
  large-finite BatchNorm outputs finite but changed moving mean/variance by
  normalizing the input before the operator; it was removed and the native CPU
  BatchNorm variance path now accumulates moments in double. Python
  `_stable_axis_norm` bypassed native LayerNorm axis validation for invalid
  negative axes; the Gluon path now validates with `_canonical_axis`. The
  recent no-affine BatchNorm/SyncBatchNorm graph workaround fixed imperative
  cases but initially missed cached-hybrid graphs; recorded CachedOp calls now
  preserve those graphs by anchoring disabled BatchNorm beta inputs with zero
  temporary gradients.
- Latent old bugs found and fixed: CPU and oneDNN BatchNorm mutated input
  `gamma` for `fix_gamma=True`; SyncBatchNorm accepted unsupported fp16 and
  failed later with an internal TBlob mismatch; cuDNN BatchNorm inference did
  not populate visible `mean`/`var`; native CUDA BatchNorm inference tied stats
  writes to a vectorized data loop and could miss channels or touch the tensor
  tail; CPU last-axis plus generic/GPU LayerNorm returned zeros when only
  visible mean/std were requested; CPU/GPU GroupNorm had the same
  primary-output `kNullOp` early-return bug for visible mean/std requests;
  oneDNN LayerNorm selected the oneDNN path for rank-5 stats tensors and
  zero-leading-dimension inputs it could not handle; SyncBatchNorm reused
  same-key CPU shared buffers across incompatible channel shapes; and
  InstanceNorm accepted non-float32 inputs until the hard-coded `real_t` compute
  path failed with a generic TBlob dtype mismatch. Regression coverage was
  added under
  `test_similar_*` in `tests/python/unittest/test_apache_open_issue_repros.py`.
  After rebuilding `libmxnet.so` from current sources, focused LayerNorm
  stats-only CPU/GPU verification passed with 4 passed, 238 deselected, and 2
  warnings; focused GroupNorm stats-only CPU/GPU verification passed with 2
  passed, 242 deselected, and 2 warnings; the oneDNN LayerNorm / SyncBatchNorm
  shape-reuse / InstanceNorm dtype validation slice passed with 5 passed, 244
  deselected, and 2 warnings; the broader normalization repro slice passed with
  35 passed, 2 xfailed, 212 deselected, and 2 warnings. A targeted cuDNN
  BatchNorm backward high-level autograd probe with only gamma grad requested
  passed, and compute-sanitizer reported 0 errors, so that audit candidate is
  not counted as verified. Current repro collection count is 249 tests. The
  full repro suite against the rebuilt current library passed with 237 passed,
  12 xfailed, and 5 warnings in 394.54s. After the CachedOp beta-anchor fix,
  focused no-affine BatchNorm/SyncBatchNorm graph verification passed normally
  with 3 passed, 246 deselected, and 2 warnings; the follow-up full repro
  suite passed with 239 passed, 10 xfailed, and 5 warnings in 393.67s.
- Latent old bugs found but not yet fixed: none in the normalization graph bucket;
  the remaining strict xfails are NumPy view/stride contract cases.


## Transformer Optional-Gradient Audit

2026-06-12 source/runtime repros confirmed that CPU transformer contrib
backward kernels returned early when the first input gradient request was
`kNullOp`, even when the second input gradient was requested. The verified
cases were `_contrib_interleaved_matmul_selfatt_valatt` skipping attention-map
gradients and `_contrib_interleaved_matmul_encdec_qk` skipping key/value
gradients. Regression tests were added under `test_similar_transformer_*`.
They failed before the C++ fix with zero second-input gradients, then passed
after rebuilding `build/libmxnet.so`: 2 passed, 249 deselected, and 2 warnings.
Existing transformer attention tests also passed with 2 passed, 1113 deselected,
and 2 warnings. A direct GPU probe for the same two optional-gradient cases
already produced the expected second-input gradients, so the verified fix is CPU
only. The full Apache repro suite then passed with 241 passed, 10 xfailed, and
5 warnings in 393.63s. The fix in `src/operator/contrib/transformer.cc` only
returns when both output requests are null and reads output TBlob pointers
inside the matching `req` branches.

## Extension OptimizeFor Backend Audit

2026-06-12 runtime verification promoted issue #19655 out of the source-only
bucket. A custom extension operator `issue19655_sleep_fill` was added to
`example/extensions/lib_subgraph/subgraph_lib.cc` so it fills an NDArray after a
sleep, and a custom partitioner `issue19655_reader` checks the first supplied
`optimize_for` argument during partition review. The new regression test
`test_optimize_for_waits_for_extension_backend_args` failed before the fix because
`MXOptimizeForBackend` exposed the pending NDArray's raw data pointer to the
extension backend without `WaitToRead()`. The C API now waits on non-null args and
aux arrays before subgraph backends or graph passes can consume those pointers.
The focused repro passed after rebuilding: 1 passed, 5 deselected, and 2 warnings
in 1.73s.

The adjacent extension sweep also found that the existing `test_subgraph` failed
for dynamically loaded graph passes: Python rejected `backend="addInputPass"`
with the static subgraph-backend precheck even though the extension had
registered the graph pass and the C++ `MXOptimizeForBackend` path supports it.
`Symbol.optimize_for` now lets the C++ backend/graph-pass lookup decide. The
focused extension pair then passed with 2 passed, 4 deselected, and 3 warnings;
the full `test_extensions.py` file passed with 4 passed, 2 skipped, and 3
warnings. The full Apache repro suite against the rebuilt current library then
passed with 241 passed, 10 xfailed, and 5 warnings in 394.63s.

## Non-Apache Xfail Sweep

2026-06-12 non-Windows xfail review found three locally addressable buckets
outside the Apache repro file: issue #18575 in control-flow, GPU quantization
xfails, and the remaining Apache NumPy view/stride xfails. The Blackwell-only
large-channel convolution xfail is not reproducible on this machine; the GPUs are
RTX 4090 / compute capability 8.9, not sm_120.

The #18575 control-flow xfail failed under `--runxfail` with an autograd safety
check while recording the outer `_cachedop`: native while-loop bookkeeping had
already attached autograd metadata to the outer `_while_loop` outputs. The fix in
`src/operator/control_flow.cc` disables autograd recording around the legacy
while-loop condition/bookkeeping copies while preserving body recording through
`LoopState::Forward`. Focused verification passed under `--runxfail`, then passed
normally after removing the marker; the full control-flow file passed with 6
passed and 2 warnings.

For GPU quantization, `test_calibrated_quantize_v2_bfloat16_to_int8` was a stale
xfail and already passed. `test_quantized_reshape` was fixed by adding GPU
registration and a device-templated quantized reshape implementation that copies
int8/uint8 payloads and range scalars on the active device. `test_quantized_transpose`
was fixed by sharing the CPU type/shape wrapper with a GPU registration that calls
the existing GPU transpose kernels and copies min/max scalars on the active device.
The six GPU uint8 quantize/quantize_v2/requantize xfails were fixed by removing
obsolete GPU-only fatal guards in the shared device-templated kernels; the same
kernels and CUDA registrations already supported the affine uint8 math and range
copy paths. Focused verification for each promoted bucket passed under
`--runxfail`, then passed normally after marker removal. The full GPU quantization
wrapper file now reports 49 passed, 4 xfailed, and 4 warnings; the same file under
`--runxfail` reports 4 failed, 49 passed, and 4 warnings, confirming there are no
stale GPU quantization xfails left in that wrapper. The remaining 4 xfails are
separate work items: GPU quantized elemwise_mul, Python `quantize_model` GPU
min/max device handling, and GPU RNN quantization / quantized_rnn support.

## Similar-Bug Sweep Repros

These tests were added after the original open-issue ledger, before any fixes
for the newly found patterns. They are strict xfails under the `test_similar_*`
namespace in `tests/python/unittest/test_apache_open_issue_repros.py`.

- Generated symbol/image validation gaps: 25 cases for symbol-side
  `box_encode`, `SequenceMask`, `SequenceLast`, `SequenceReverse`,
  `arange_like`, self-attention heads, image resize/random crop, NDArray/npx
  random resized crop validation, and high-level `mx.image` validation.
- Mixed-device wrapper gaps: 44 NumPy public-wrapper cases and 27 legacy
  NDArray cases that still need a same-device copy or rejection path.
- Gluon repeated-pattern bugs: 3 no-affine BatchNorm/SyncBatchNorm graph
  cases, 3 hybrid CPU RNN sequence-length caching cases, normalization
  large-finite overflow cases, CosineEmbeddingLoss and other Gluon loss
  numeric-edge cases, 3 dynamic_unroll int32 valid_length cases, and one
  native LP-pooling large-finite overflow case.
- Sparse/view repeated-pattern bugs: CSR, CSR-dense, row_sparse retain,
  row_sparse elemwise, sparse unary/scalar canonicalization cases, plus
  stepped-slice, axis movement, reshape-like, flip/rot/squeeze/atleast_* NumPy
  view-contract cases.
- Verification after adding the first wave: `pytest -q ... -k test_similar`
  reported 46 xfailed, 53 deselected, and 5 warnings in 200.85s. Verification
  after adding the second wave and dropping the unverified oneDNN FC branch
  candidate reported 64 xfailed, 78 deselected, and 2 warnings in 300.97s for
  the selected second-wave slice. Focused verification after fixing InstanceNorm
  non-default-axis deferred shape inference and Gluon loss numeric-edge cases
  passed with 10 passed, 132 deselected, and 2 warnings in 1.11s under
  `--runxfail`. The recursive Halley/Sagan batch then verified with 66 xfailed,
  142 deselected, and 2 warnings in 422.40s; the recursive Anscombe/Aristotle
  batch verified with 19 xfailed, 208 deselected, and 2 warnings in 2.59s.
  Parent-checkout verification of the current promotion batch reported 15 passed, 212 deselected for validation wrappers; 3 passed, 217 deselected, 7 xfailed for the generated validation-wrapper split; 5 passed, 212 deselected, 10 xfailed for view-contract cases; 71 passed, 156 deselected for cross-device wrappers; and 4 passed, 209 deselected, 14 xfailed for the partial numeric-stability promotion. Focused dynamic_unroll int32 valid_length verification passed with 3 passed, 224 deselected, and 2 warnings in 0.49s under --runxfail. After removing the xfail marker, the same focused slice passed normally with 3 passed, 224 deselected, and 2 warnings in 0.48s. Focused loss/cosine numeric verification passed with 14 passed, 213 deselected, and 2 warnings in 1.43s under --runxfail; after removing the xfail markers, the same focused slice passed normally with 14 passed, 213 deselected, and 2 warnings in 1.59s. Focused GroupNorm large-finite verification passed with 2 passed, 225 deselected, and 2 warnings in 0.31s under --runxfail; after removing the xfail marker, the same focused slice passed normally with 2 passed, 225 deselected, and 2 warnings in 0.34s. Focused no-affine BatchNorm verification now passes normally with 3 passed, 246 deselected, and 2 warnings after the recorded CachedOp beta-anchor fix. Focused wrapper/SequenceLast/SequenceReverse verification used the installed wheel library because build/libmxnet.so was absent: the promoted subset passed under --runxfail with 13 passed, 214 deselected, and 2 warnings in 38.66s; the full focused slice then passed normally with 13 passed, 213 deselected, 1 xfailed, and 2 warnings in 42.00s. Focused hybrid CPU RNN sequence-length verification passed with 3 passed, 224 deselected, and 2 warnings in 0.50s under --runxfail; after removing the xfail marker, the same focused slice passed normally with 3 passed, 224 deselected, and 2 warnings in 0.47s. Full repro checkpoint against local Python sources plus the installed wheel library passed with 193 passed, 34 xfailed, and 3 warnings in 380.45s before removing the documented mx.image.random_crop interp=10 non-bug candidate from the bug repro count. After that removal, the generated-wrapper validation slice passed with 9 passed, 217 deselected, and 2 warnings in 26.76s. Focused BatchNorm/SyncBatchNorm large-finite verification now passes normally with 4 passed, 222 deselected, and 2 warnings in 0.79s. Focused native sparse/storage verification against build/libmxnet.so passes normally with 11 passed, 215 deselected, and 2 warnings in 1.57s. Focused LP pooling verification against build/libmxnet.so passed under --runxfail with 1 passed, 225 deselected, and 2 warnings in 0.20s, and then passed normally with 1 passed, 225 deselected, and 2 warnings in 0.23s. Current audited focused verification passed after rebuilding: static-memory pair 2 passed; scalar-backward/simple_bind/static-shape repros 3 passed; Gluon trainer 1 passed; NCCL updater 1 passed. BatchNorm/LayerNorm audit verification passed after rebuilding: 8 focused CPU repros passed, 2 focused GPU BatchNorm inference-stat repros passed, and compute-sanitizer reported 0 errors for the native CUDA small-tail inference case. Collection count is now 251 total repro tests, 241 normal, and 10 xfailed after adding and fixing the CPU transformer optional-gradient cases.


## Build/Test Sweep Checkpoint

2026-06-12 clean rebuild and wheel sweep checkpoint:

- Clean CMake rebuild completed for wheel tag `2.0.0+cu13.bw.20260609.1`
  with CUDA, cuDNN, NCCL, oneDNN, OpenCV, int64 tensor sizes, and the CUDA
  architectures used by the latest `smolix/mxnet` wheel. The built artifact is
  `dist/mxnet-2.0.0+cu13.bw.20260609.1-cp312-cp312-linux_x86_64.whl`.
- C++ unit launcher bug found and fixed: `tests/run_unit_test_shards.sh.in`
  captured `$?` after the surrounding `if` statement, so a failed shard printed
  `[FAIL] ... (exit 0)` and the launcher could still report success. The
  template now captures the failing executable status in the `else` branch.
  Verification: `build/tests/mxnet_unit_tests --gtest_filter=Engine.RandSumExpr`
  now exits 1 and reports `[FAIL ] engine (exit 1)`.
- Native C++ suite-found failure fixed: `Engine.RandSumExpr` was throwing from
  `src/engine/threaded_engine.cc:287` with `duplicate items found in const_vars`
  because direct threaded-engine callers could pass duplicate dependency handles
  that C API/imperative callers already normalized. `ThreadedEngine::NewOperator`
  now deduplicates dependency vectors before storing/checking them, and
  `Engine.ThreadedPushAsyncDeduplicatesDirectDependencies` covers duplicate
  direct read/write dependencies. Verification: the focused engine binary passed
  `Engine.RandSumExpr` plus the new regression test, and
  `build/tests/mxnet_unit_tests --gtest_filter=Engine.RandSumExpr` reported all
  shards passed.
- Wheel acceptance harness updated so a `cp312` wheel creates a Python 3.12 venv
  automatically and stores scratch/report/cache data under `.tmp/` by default.
  The installed wheel import check passed and reported 4 GPUs plus CUDA, cuDNN,
  NCCL, oneDNN, and OpenCV enabled.
- Wheel sweep status before stopping a hung shard: `cpu_xop19`,
  `cpu_optimized_validation`, `cpu_optimizer`, `cpu_gluon_parameter`,
  `cpu_layer_norm`, and `cpu_group_norm` passed. The broad `cpu_unittest` shard
  stopped advancing at 98% and was terminated as hung after recording 145 unique
  failed/error test ids in `.tmp/wheel-test-20260612T034239Z/shards/cpu_unittest.log`.
  The partial failures cluster as: 107 NumPy operator failures, 8 Gluon failures,
  7 OpenCV/image failures, 4 tricky GPU index-update/add failures, 3 trainer
  failures, 2 profiler failures, 2 concurrency/lifetime failures, 2 subgraph
  failures, and single failures/errors in apache-open-issue repro, deferred
  compute, exception handling, control flow, metric, ndarray order, NumPy
  interoperability, NumPy loss, NumPy ndarray indexing, sparse model load, and
  sparse ndarray tests. These are suite-found failures pending focused reruns and
  shared-root-cause triage; they are not yet added to the runtime-verified bug
  repro count.
- Current-code Python sweep follow-up: `tools/run_pytest_limited_threads.py`
  now defaults to importing `/home/smola/mxnet/python`, loading
  `/home/smola/mxnet/build/libmxnet.so`, and exporting the same `PYTHONPATH` for
  subprocess-backed tests. This prevents focused pytest runs from accidentally
  testing a stale installed wheel against the latest native library.
- Suite-found NumPy failures fixed or resolved against latest code:
  `test_np_squeeze` now preserves autograd/deferred-compute graph edges for
  no-op squeezes and handles literal zero dimensions through NumPy reshape;
  scalar-input `np.unravel_index` now returns 0-D MXNet NumPy NDArrays instead
  of Python ints; `test_np_bitwise_shift` now follows the shared ufunc wrapper
  contract by expecting `NotImplementedError` for legal but unsupported kwargs.
  Verification: the focused seven-test sample passed, and the full
  `test_np_unravel_index` plus `test_np_bitwise_shift` parameterized selection
  passed with 446 passed and 2 warnings in 53.15s.
- Focused reruns against latest code narrowed the still-reproduced Python
  failures to static-memory backward shape metadata
  (`shape_num_unknown_nodes != 0`), trainer updater state placement
  (`cpu_pinned(0)` vs `cpu(0)` for `sgd_mom_update`), and the static-shape
  subgraph paramless-data binding repro. This audited batch fixes those paths and
  adds/keeps focused repro coverage for scalar backward shape preservation,
  dynamic-output `simple_bind`, static-shape paramless data, local/NCCL updater
  context alignment, and `test_hybrid_static_memory[True-True]`. Representative
  CachedOp/cuDNN BatchNorm, OpenCV/image, sparse, profiler, GPU index-bounds,
  and optimizer hang-suspect selections passed in the current-code rerun.

## Runtime/API-Verified Repros

| GitHub item | Test | Current symptom |
|---|---|---|
| PR #21217 | `test_pr_21217_horovod_kvstore_exposes_barrier` | Fixed in current worktree; Horovod KVStore exposes `_barrier()`. |
| issue #21176 | `test_issue_21176_conv2d_nhwc_cpu_runs` | Fixed in current worktree; CPU NHWC `Conv2D` runs by dispatching through NCHW on CPU. |
| PR #21044 | `test_pr_21044_symbolblock_preserves_symbol_parameter_attrs` | Fixed in current worktree; now preserves user `lr_mult`, `wd_mult`, and initializer attributes. |
| issue #21119 | `test_issue_21119_cross_gpu_binary_op_does_not_hang` | Fixed in current worktree; public NumPy binary wrappers reject operands on different devices before backend dispatch. |
| issue #21111 | `test_issue_21111_cudnn_batchnorm_cachedop_forward_only_train_mode_is_stateless` | Fixed in current worktree; non-recording train-mode `CachedOp` calls copy mutable aux states so BatchNorm cannot leak forward-only state mutations. |
| issue #21156 | `test_issue_21156_indexed_recordio_close_survives_module_teardown` | Fixed in current worktree; close no longer depends on the module global and tolerates partial teardown state. |
| issue #21146 | `test_issue_21146_gru_deferred_init_with_sequence_length_runs` | Fixed in current worktree; GRU sequence lengths are passed by keyword and non-LSTM state is returned as a tensor. |
| issue #20936 | `test_issue_20936_wheel_exposes_include_path` | Fixed in current worktree; `find_include_path()` returns include paths as a list. |
| issue #20657 | `test_issue_20657_find_conf_path_env_override_is_sequence` | Fixed in current worktree; env override now returns a one-item list. |
| issue #20605 | `test_issue_20605_csr_gradient_preserves_sparse_pattern` | Fixed in current worktree; CSR dot gradients preserve the source CSR sparse pattern. |
| issue #20577 | `test_issue_20577_symbolblock_export_succeeds_without_cached_op_args` | Fixed in current worktree; export falls back to collected params when cached-op args are absent. |
| issue #20391 | `test_issue_20391_numpy_gluon_allows_row_sparse_gradients` | Fixed in current worktree; sparse data/grad buffers use legacy sparse NDArrays under NumPy mode. |
| PR #20491 | `test_pr_20491_cpp_symbol_exposes_optimize_for_backend` | Fixed in current worktree; C++ `Symbol` exposes `OptimizeForBackend`. |
| issue #20037 | `test_issue_20037_recordio_preserves_large_integer_label` | Fixed in current worktree; scalar labels not exactly representable as float32 round-trip via a float64 payload extension. |
| issue #20180 | `test_issue_20180_box_encode_zero_refs_is_validated_or_empty` | Fixed in current worktree; empty refs are rejected before the internal `TBlob` path. |
| issue #20076 | `test_issue_20076_sequence_mask_rejects_huge_lengths_cleanly` | Fixed in current worktree; out-of-range `sequence_length` values are rejected in Python. |
| issue #20046 | `test_issue_20046_image_resize_invalid_interp_has_mxnet_validation` | Fixed in current worktree; invalid interpolation ids are rejected before OpenCV. |
| issue #20044 | `test_issue_20044_boolean_mask_empty_out_is_safe` | Fixed in current worktree; empty `boolean_mask` input is rejected before async execution. |
| issue #19860 | `test_issue_19860_swish_negative_beta_zero_input_is_finite` | Fixed in current worktree; zero Swish inputs bypass the unstable extreme-beta sigmoid path. |
| issue #19852 | `test_issue_19852_instancenorm_large_finite_input_is_finite` | Fixed in current worktree; imperative InstanceNorm uses a float64 variance path for large finite inputs. |
| issue #19785 | `test_issue_19785_groupnorm_zero_groups_is_python_error_not_abort` | Fixed in current worktree; `GroupNorm(num_groups=0)` raises a Python `ValueError`. |
| issue #19753 | `test_issue_19753_topk_indices_are_integer_typed` | Fixed in current worktree; `topk` index outputs are returned with integer dtype. |
| issue #19628 | `test_issue_19628_gpu_ctcloss_accepts_fp16_predictions` | Fixed in current worktree; GPU `CTCLoss` accepts FP16 predictions without the internal dtype mismatch. |
| issue #19659 | `test_issue_19659_hybrid_boolean_mask_backward_runs` | Fixed in current worktree; cached-op backward allows zero-output-gradient subgraphs that depend only on saved inputs/outputs. |
| issue #19655 | `test_optimize_for_waits_for_extension_backend_args` | Fixed in current worktree; `MXOptimizeForBackend` waits for supplied args/aux before extension backends or graph passes receive raw data pointers. |
| issue #19686 | `test_issue_19686_selfatt_qk_rejects_zero_heads_cleanly` | Fixed in current worktree; zero attention heads are rejected before backend dispatch. |
| issue #19683 | `test_issue_19683_arange_like_repeat_zero_is_safe` | Fixed in current worktree; non-positive `repeat` is rejected before backend dispatch. |
| issue #19647 | `test_issue_19647_optimize_for_missing_backend_raises` | Fixed in current worktree; missing optimization backends raise instead of returning a symbol after logging. |
| issue #19423 | `test_issue_19423_choice_full_without_replacement_is_permutation` | Fixed in current worktree; full-range no-replacement choice now produces a non-identity permutation for some seeds. |
| issue #19458 | `test_issue_19458_tensordot_scalar_empty_axes_backward` | Fixed in current worktree; scalar empty-axis `tensordot` backward returns finite correct gradients. |
| issue #19422 | `test_issue_19422_numpy_array_iteration_yields_python_scalars` | Fixed in current worktree; NumPy ndarray iteration yields Python scalars for scalar elements. |
| issue #19170 | `test_issue_19170_stepped_slice_shares_storage` | Still blocked: stepped NumPy slicing needs non-unit stride metadata in ndarray/view handles; current `_npi.slice` materializes a dense copy. Shape-only view helpers (`ravel`, `squeeze`, `atleast_*`) are fixed in Python via `reshape_view`. |
| PR #18583 | `test_pr_18583_cpp_symbol_exposes_partial_shape_inference` | Fixed in current worktree; C++ `Symbol` exposes partial shape inference. |
| issue #19021 | `test_issue_19021_backward_rejects_mismatched_head_gradient_shape` | Fixed in current worktree; Python backward rejects head gradients with mismatched shapes. |
| issue #18919 | `test_issue_18919_numpy_advanced_indexing_matches_numpy` | Fixed in current worktree; mixed advanced index arrays are broadcast before indexing. |
| issue #18770 | `test_issue_18770_non_native_byte_order_is_not_silently_lost` | Fixed in current worktree; non-native byte order inputs are rejected instead of silently normalized. |
| PR #18792 | `test_pr_18792_sort_and_argsort_support_float16` | Fixed in current worktree; legacy `sort`/`argsort` handle float16 inputs through a float32 dispatch path. |
| issue #18669 | `test_issue_18669_zoneout_output_matches_new_state` | Fixed in current worktree; `ZoneoutCell` returns output consistent with its first recurrent state. |
| issue #18563 | `test_issue_18563_max_backward_splits_tied_gradient` | Fixed in current worktree; tied full-reduction extrema split gradient across equal winners. |
| issue #18078 | `test_issue_18078_prod_backward_multiple_zeros_is_finite` | Fixed in current worktree; full-reduction `prod` with multiple zeros returns finite zero gradients. |
| issue #18300 | `test_issue_18300_numpy_prod_accepts_shape_tuple` | Fixed in current worktree; public `mxnet.numpy.prod` converts array-like input with `asarray`. |
| PR #17209 | `test_pr_17209_parameter_symbol_var_omits_dtype_attribute` | Fixed in current worktree; `Parameter.var()` no longer emits a fixed dtype attribute. |
| issue #17936 | `test_issue_17936_gammaln_promotes_integer_input` | Fixed in current worktree; integer inputs to `npx.gammaln` are promoted before evaluation. |
| issue #17698 | `test_issue_17698_split_and_load_does_not_materialize_full_input_first` | Fixed in current worktree; NumPy inputs are split before per-context MXNet materialization. |
| issue #11774 | `test_issue_11774_batchnorm_without_scale_or_center_trains` | Fixed in current worktree; `BatchNorm(scale=False, center=False)` preserves the training autograd graph. |
| issue #16402 | `test_issue_16402_legacy_ndarray_dtype_is_numpy_dtype_object` | Fixed in current worktree; legacy `NDArray.dtype` now returns a `numpy.dtype`. |
| issue #16427 | `test_issue_16427_recordio_pack_accepts_python3_string_payload` | Fixed in current worktree; `recordio.pack()` encodes Python 3 string payloads before concatenation. |
| issue #13953 | `test_issue_13953_upsampling_accepts_data_keyword` | Fixed in current worktree; vararg Symbol wrappers map `data`/`weight` Symbol kwargs to backend `arg0`/`arg1`. |
| issue #13945 | `test_issue_13945_indexed_recordio_shared_reader_is_thread_safe` | Fixed in current worktree; indexed reads and writes are guarded by a per-reader lock. |
| issue #13193 | `test_issue_13193_sparse_elemwise_mul_has_canonical_csr_payload` | Fixed in current worktree; CSR sparse `elemwise_mul` output is canonicalized before return. |
| issue #8430 | `test_issue_8430_ndarrayiter_preserves_integer_label_dtype` | Fixed in current worktree; `NDArrayIter` preserves NumPy label dtype during construction and shuffle. |
| issue #12286 | `test_issue_12286_ndarray_wrapper_raises_python_typeerror_for_missing_inputs` | Fixed in current worktree; generated NDArray wrappers translate backend input-count mismatches to `TypeError`. |
| issue #8817 | `test_issue_8817_sparse_zeros_accepts_integer_shape` | Fixed in current worktree; sparse zeros normalizes integer shapes to one-dimensional tuples. |
| issue #14695 | `test_issue_14695_single_output_ndarray_is_not_tuple_unpackable` | Fixed in current worktree; legacy split/SliceChannel remain list-returning even with one output. |

## Source-Verified Pending Runtime Repro

These were source-verified in the scan but still need an executable repro before
being promoted to the table above.

Issue-side source/static-only candidates still pending a runtime repro:
`#20376`.

PR-side source-verified pending runtime/static API repro:

`#20470` and `#20316`. The original PR candidate set and the follow-up
non-obvious-title PR candidates are annotated below by current status.

## Follow-Up Broad Scan Notes

This pass waited 10 minutes, then re-scanned open items outside the original
`Bug`/`Flaky` issue slice: unlabeled/non-bug-labeled issues, GitHub items that
reference Discuss/JIRA/StackOverflow/GitHub Discussions, and non-obvious open
PR titles. Four agents were used for parallel triage. The external-reference
agent covered 1,804 open issues, 203 open PRs, and 1,256 open non-`Bug`/`Flaky`
items. The PR-title agent screened 123 non-obvious-title PRs, but no additional
PR was promoted without a runtime or static failure in this checkout.

Newly promoted executable repros from the broad/retry pass:

- `#20577`, `#8430`, `#19458`, `#16427`, `#13953`, `#13945`, `#13193`,
  `#11774`, `#18300`, `#12286`, and `#8817`.
- `#13945` also covers the still-open attempted fix PR `#18366`.
- `#17951` is a linked float16 sort/argsort request covered by the existing
  PR `#18792` repro.

Retried but not promoted because the current wheel appears fixed, the affected
API path is gone, or the symptom was not deterministic enough for a credible
unit repro:

- NumPy compatibility/operator retries: `#21165`, `#20886`, `#20880`.
- Runtime-crash retries: `#20005`, `#20842`, `#16936`, `#16051`.
- Older API/operator retries: `#17088`, `#16855`, `#16745`, `#15079`,
  `#13909`, `#11551`, `#8785`, `#9159`.
- GPU memory report `#20315`: the exact loop plateaued in this wheel in both
  the `asnumpy()` and no-`asnumpy()` fresh-process comparisons, so it is not a
  current deterministic leak repro here.


Similar-sweep candidates tried but not promoted:

- oneDNN FC branch subgraph candidate: a proposed repro XPASSed in this
  checkout, so it was removed from the test file and is not counted as a
  verified current bug.

Broad-scan items kept out of xfail tests:

- `#8219` is a real performance concern from a JIRA-linked report, but it needs
  a benchmark/perf guard rather than a normal unit xfail.
- `#12062` is distributed ps-lite hostname behavior; the local checkout lacks
  populated `3rdparty/ps-lite`, and no distributed repro was run.
- `#15215` is an old sparse/distributed embedding workflow. The current wheel no
  longer exposes the old `gluon.contrib.nn.SparseEmbedding` API used by the
  report, and current source still explicitly rejects incomplete sparse SGD row
  updates.
- Platform/build/binding reports `#20766`, `#21154`, and `#20844` were not
  verified in this Python wheel environment.
- Packaging/request-like items `#21226`, `#21210`, `#20336`, `#20224`,
  `#20118`, and `#20147` were not counted as confirmed local code bugs.

Unconfirmed PR candidates from the non-obvious-title PR pass:

`#18325`, `#18285`, `#21091`, `#17754`, `#18928`, `#15996`, `#15994`,
`#15993`, `#15811`, `#20249`, `#21215`, `#20685`, `#20569`, `#19849`,
`#18678`, `#19646`, `#14911`, `#21212`.

Notes:

- `#18325` and `#18285` are optimizer semantic changes linked to `#15533`.
- `#21091` and `#17754` are feature/API support for non-leaf gradients and
  higher-order gradients, not confirmed current bugs.
- `#18928`, `#15996`, `#15994`, `#15993`, `#15811`, and `#20249` are ONNX,
  quantization, or operator capability work requiring dedicated artifacts or
  GPU/ONNX coverage before promotion.
- `#21215`, `#20685`, `#20569`, `#19849`, `#18678`, and `#19646` are build,
  dependency, CUDA/TRT, or packaging work.
- `#14911` is an engine callback API cleanup candidate, and `#21212` is a
  test-only PR; neither was locally verified as a failing behavior.

## Full Annotated Inventory

The exhaustive scan inventory is retained here so each GitHub item can be
updated as runtime verification proceeds.

### Issues: Runtime-Verified

`#21176`, `#21119`, `#21111`, `#21156`, `#21146`, `#20936`, `#20657`,
`#20605`, `#20577`, `#20391`, `#20180`, `#20076`, `#20046`, `#20044`,
`#20037`, `#19860`, `#19852`, `#19785`, `#19753`, `#19686`, `#19683`,
`#19659`, `#19655`, `#19647`, `#19628`, `#19458`, `#19423`, `#19422`, `#19170`,
`#19021`, `#18919`, `#18770`, `#18669`, `#18575`, `#18563`, `#18300`, `#18078`,
`#17936`, `#17698`, `#16427`, `#16402`, `#13953`, `#13945`, `#13193`,
`#12286`, `#11774`, `#8817`, `#8430`, `#14695`.

### Issues: Source-Verified Only

`#20376`.

Notes:

- `#20376` is covered by the TensorRT/ONNX conversion source issue also tracked
  as PR `#20470`; the current wheel has no TensorRT runtime, so this remains
  source/static only.

### Issues: Static, Documentation, Or API Surface Verified

`#20625`, `#20010`, `#19080`, `#18668`, `#8219`.

Notes:

- `#20625` and `#20010` are documentation/Doxygen issues, not runtime
  failures.
- `#19080` is a C API usability gap: `MXEnginePush*` still take
  `ContextHandle`, while the public C API otherwise exposes device type/id
  integers.
- `#18668` is a C++ API dtype inference issue in `InferArgsMap`; it is source
  visible in `cpp-package/include/mxnet-cpp/symbol.hpp` but was not converted
  into a Python runtime repro.
- `#8219` is a JIRA-linked broadcast performance issue. It reproduced as a
  large timing gap in a local probe, but it needs benchmark infrastructure
  rather than an xfail unit test.

### Issues: Fixed Or Not Current In This Fork

`#21225`, `#21199`, `#21190`, `#21153`, `#21143`, `#21084`, `#20968`,
`#20951`, `#20886`, `#20880`, `#20875`, `#20870`, `#20842`, `#20824`, `#21165`, `#20784`, `#20729`,
`#20769`, `#20659`, `#20651`, `#20639`, `#20467`, `#20460`, `#20447`, `#20440`,
`#20411`, `#20315`, `#20282`, `#20223`, `#20197`, `#20183`, `#20182`, `#20181`,
`#20128`, `#20123`, `#20079`, `#20064`, `#20062`, `#20052`, `#20051`,
`#20050`, `#20049`, `#20047`, `#20045`, `#20041`, `#20040`, `#20039`,
`#20005`, `#19991`, `#19941`, `#19921`, `#19907`, `#19891`, `#19859`, `#19825`,
`#19798`, `#19793`, `#19784`, `#19777`, `#19609`, `#19495`, `#19477`,
`#19369`, `#19353`, `#19343`, `#19252`, `#19084`, `#19030`, `#18944`,
`#18940`, `#18918`, `#18866`, `#18865`, `#18791`, `#18789`, `#18600`,
`#18398`, `#18171`, `#18117`, `#17988`, `#17913`, `#17850`, `#17661`,
`#17218`, `#17088`, `#16936`, `#16855`, `#16851`, `#16828`, `#16745`,
`#16591`, `#16051`, `#15988`, `#15383`, `#15079`, `#14710`, `#14264`,
`#14227`, `#13909`, `#13485`, `#12389`, `#11865`, `#11551`, `#11384`,
`#11032`, `#10494`, `#10045`, `#9159`, `#8785`.

### Issues: Environment, Support, Platform, Or Not A Local Code Bug

`#21209`, `#21208`, `#21189`, `#21187`, `#21179`, `#21178`, `#21170`,
`#21138`, `#21135`, `#21125`, `#21109`, `#21085`, `#21081`, `#21069`,
`#21035`, `#20985`, `#20954`, `#20945`, `#20901`, `#20898`, `#20885`,
`#20845`, `#20758`, `#20733`, `#20687`, `#20671`, `#20656`, `#20483`,
`#20469`, `#20422`, `#20416`, `#20405`, `#20390`, `#20343`, `#20329`,
`#20307`, `#20286`, `#20256`, `#20217`, `#20143`, `#20134`, `#20081`,
`#19949`, `#19943`, `#19781`, `#19731`, `#19717`, `#19651`, `#19649`,
`#19619`, `#19591`, `#19583`, `#19580`, `#19550`, `#19436`, `#19420`,
`#19351`, `#19211`, `#19144`, `#19111`, `#19088`, `#19082`, `#19005`,
`#19003`, `#19002`, `#19001`, `#19000`, `#18999`, `#18991`, `#18990`,
`#18989`, `#18985`, `#18962`, `#18960`, `#18957`, `#18898`, `#18869`,
`#18860`, `#18855`, `#18833`, `#18832`, `#18831`, `#18808`, `#18774`,
`#18764`, `#18759`, `#18739`, `#18729`, `#18726`, `#18716`, `#18693`,
`#18657`, `#18641`, `#18638`, `#18628`, `#18592`, `#18590`, `#18551`,
`#18514`, `#18509`, `#18501`, `#18481`, `#18468`, `#18449`, `#18436`,
`#18433`, `#18430`, `#18428`, `#18417`, `#18396`, `#18389`, `#18321`,
`#18305`, `#18278`, `#18276`, `#18262`, `#18258`, `#18255`, `#18231`,
`#18227`, `#18217`, `#18216`, `#18215`, `#18214`, `#18192`, `#18191`,
`#18163`, `#18153`, `#18124`, `#18121`, `#18108`, `#18073`, `#18048`,
`#18013`, `#17978`, `#17943`, `#17942`, `#17938`, `#17920`, `#17887`,
`#17874`, `#17855`, `#17848`, `#17847`, `#17845`, `#17806`, `#17774`,
`#17729`, `#17726`, `#17723`, `#17720`, `#17686`, `#17680`, `#17665`,
`#17662`, `#17627`, `#17621`, `#17588`, `#17581`, `#17518`, `#17483`,
`#17470`, `#17469`, `#17461`, `#17459`, `#17439`, `#17436`, `#17395`,
`#17394`, `#17380`, `#17347`, `#17315`, `#17310`, `#17291`, `#17282`,
`#17260`, `#17258`, `#17257`, `#17256`, `#17250`, `#17246`, `#17231`,
`#17221`, `#17207`, `#17205`, `#17197`, `#17145`, `#17136`, `#17108`,
`#17092`, `#17081`, `#17080`, `#17079`, `#17076`, `#17046`, `#17045`,
`#17043`, `#17033`, `#16988`, `#16983`, `#16963`, `#16933`, `#16904`,
`#16880`, `#16863`, `#16803`, `#16741`, `#16675`, `#16620`, `#16539`,
`#16499`, `#16456`, `#16449`, `#16441`, `#16326`, `#16210`, `#16193`,
`#16045`, `#15997`, `#15892`, `#15790`, `#15789`, `#15540`, `#15326`,
`#15297`, `#14967`, `#14263`, `#14203`, `#14087`, `#13518`, `#13342`,
`#13314`, `#12799`, `#12472`, `#11565`, `#11542`, `#11163`, `#10737`,
`#10004`, `#9967`, `#9572`, `#9271`, `#9096`, `#8234`, `#7933`, `#7247`.

### Issues: Attempted But Not Runtime-Verified

`#21144`, `#21118`, `#21059`, `#21052`, `#21019`, `#21005`, `#20959`,
`#20805`, `#20802`, `#20754`, `#20702`, `#20691`, `#20675`,
`#20632`, `#20471`, `#20465`, `#20394`, `#20330`, `#20317`, `#20300`,
`#20290`, `#20280`, `#20159`, `#19994`, `#19841`, `#19803`, `#19577`,
`#19574`, `#19556`, `#19498`, `#19333`, `#19231`, `#19218`, `#19159`,
`#19155`, `#19073`, `#19066`, `#19056`, `#19024`, `#19019`, `#18923`,
`#18834`, `#18806`, `#18776`, `#18751`, `#18743`, `#18699`, `#18659`,
`#18643`, `#18617`, `#18584`, `#18476`, `#18466`, `#18265`,
`#18254`, `#18253`, `#18209`, `#18198`, `#18165`, `#18135`, `#18024`,
`#17981`, `#17960`, `#17931`, `#17898`, `#17888`, `#17840`, `#17836`,
`#17833`, `#17829`, `#17814`, `#17810`, `#17782`, `#17744`, `#17703`,
`#17694`, `#17653`, `#17651`, `#17633`, `#17612`, `#17568`, `#17565`,
`#17554`, `#17522`, `#17495`, `#17493`, `#17488`, `#17480`, `#17471`,
`#17454`, `#17412`, `#17411`, `#17381`, `#17363`, `#17357`, `#17342`,
`#17335`, `#17182`, `#17144`, `#17126`, `#17106`, `#17064`, `#17062`,
`#16960`, `#16956`, `#16938`, `#16929`, `#16925`, `#16816`,
`#16806`, `#16757`, `#16752`, `#16705`, `#16701`, `#16686`, `#16685`,
`#16656`, `#16604`, `#16590`, `#16548`, `#16483`, `#16434`,
`#16365`, `#16188`, `#16187`, `#16140`, `#16134`, `#16098`, `#16093`,
`#16087`, `#16060`, `#15932`, `#15809`, `#15766`, `#15296`,
`#15283`, `#15196`, `#15125`, `#15102`, `#15067`, `#14983`, `#14975`,
`#14727`, `#14690`, `#14522`, `#14447`, `#14373`, `#14340`, `#14317`,
`#13592`, `#13341`, `#13332`, `#13264`, `#13199`,
`#13138`, `#12894`, `#12760`, `#12555`, `#12444`, `#12337`,
`#11965`, `#11794`, `#11638`, `#11314`, `#11275`, `#10489`,
`#10357`, `#10220`, `#10173`, `#8337`, `#8239`, `#7847`, `#7664`,
`#7080`, `#4887`, `#4659`.

### Issues: CI Or Flaky Infrastructure Only

`#21216`, `#20978`, `#20964`, `#20960`, `#21114`, `#21113`, `#21061`,
`#21040`, `#21006`, `#20979`, `#20778`, `#20738`, `#20529`, `#20441`,
`#20455`, `#20389`, `#20374`, `#20337`, `#20334`, `#20289`, `#20265`,
`#20239`, `#20088`, `#20011`, `#19938`, `#19915`, `#19673`, `#19636`,
`#19623`, `#19622`, `#19616`, `#19606`, `#19511`, `#19330`, `#19227`,
`#19183`, `#19166`, `#19101`, `#19081`, `#19071`, `#19007`, `#18971`,
`#18920`, `#18881`, `#18829`, `#18809`, `#18756`, `#18745`, `#18740`,
`#18732`, `#18618`, `#18564`, `#18527`, `#18442`, `#18420`, `#18400`,
`#18382`, `#18381`, `#18374`, `#18334`, `#18330`, `#18294`, `#18291`,
`#18282`, `#18233`, `#18225`, `#18210`, `#18184`, `#18175`, `#18166`,
`#18149`, `#18144`, `#18101`, `#18100`, `#18098`, `#18090`, `#18088`,
`#18086`, `#18059`, `#17954`, `#17935`, `#17731`, `#17667`, `#17666`,
`#17636`, `#17635`, `#17558`, `#17557`, `#17504`, `#17498`, `#17467`,
`#17414`, `#17397`, `#17369`, `#17219`, `#17151`, `#17067`, `#17022`,
`#16962`, `#16945`, `#16839`, `#16831`, `#16799`, `#16776`, `#16770`,
`#16739`, `#16725`, `#16674`, `#16600`, `#16566`, `#16367`, `#16359`,
`#16345`, `#16217`, `#16208`, `#16181`, `#16172`, `#16162`, `#16030`,
`#15975`, `#15925`, `#15856`, `#15786`, `#15732`, `#15603`, `#15423`,
`#15406`, `#15284`, `#15199`, `#15034`, `#14970`, `#14852`, `#14723`,
`#14719`, `#14718`, `#14555`, `#14552`, `#14524`, `#14482`, `#14366`,
`#14329`, `#14292`, `#14288`, `#14285`, `#14234`, `#14189`, `#14174`,
`#14101`, `#13958`, `#13743`, `#13577`, `#13484`, `#13439`, `#13103`,
`#12901`, `#12675`, `#12658`, `#12415`, `#11801`, `#11758`, `#11727`,
`#11726`, `#11725`, `#11724`, `#11723`, `#11720`, `#11713`, `#11707`,
`#11701`, `#11654`, `#11592`, `#11517`, `#11509`, `#11441`, `#11395`,
`#11388`, `#11290`, `#10274`, `#10141`, `#9857`, `#9845`, `#9381`.

### PRs: Runtime/API-Verified

`#21217`, `#21044`, `#20491`, `#18792`, `#18583`, `#17209`.

Covered by issue repro:

- `#18366` is the attempted RecordIO threading fix and is covered by
  `test_issue_13945_indexed_recordio_shared_reader_is_thread_safe`.

### PRs: Attempted But Not Current In This Fork

`#20814`, `#20508`, `#20454`, `#18268`, `#18125`, `#18112`, `#18027`,
`#17975`, `#17871`, `#16854`, `#14582`.

### PRs: Build, Dependency, Platform, CI, Or Documentation Only

`#21221`, `#21164`, `#21073`, `#21071`, `#20352`, `#20287`, `#20108`,
`#18977`, `#18967`, `#18418`, `#17955`, `#17917`, `#17794`, `#17693`,
`#17373`.

### PRs: Source-Only Or Inconclusive Runtime Repro

`#20470`, `#20316`.

Notes:

- `#20470` and `#20316` require TensorRT/subgraph builder coverage rather than a
  minimal Python operator repro.

### PRs: Broad-Scan Pending Or Unconfirmed

`#18325`, `#18285`, `#21091`, `#17754`, `#18928`, `#15996`, `#15994`,
`#15993`, `#15811`, `#20249`, `#21215`, `#20685`, `#20569`, `#19849`,
`#18678`, `#19646`, `#14911`, `#21212`.

Notes:

- These came from the interrupted non-obvious-title PR pass and were not counted
  as current bugs because no runtime/static failure was verified locally.
- The ONNX/TRT/build/dependency candidates need dedicated artifacts or build
  matrix coverage before promotion.

### PRs: Fixed, Superseded, Not Applicable, Or Unknown

- Fixed/superseded locally: `#21224`, `#21213`, `#21204`, `#21159`,
  `#21155`, `#20761`, `#20351`, `#20281`, `#19993`, `#19913`, `#19844`,
  `#19275`, `#19076`, `#18526`, `#18521`, `#16700`, `#15857`, `#15702`,
  `#14738`.
- Not applicable to this tree: `#21141`, `#20297`, `#19315`, `#18636`,
  `#18349`, `#17769`, `#17533`, `#15672`, `#15566`, `#13917`.
- Unknown/inconclusive: `#20158`, `#20089`, `#18615`, `#14452`, `#14320`.
