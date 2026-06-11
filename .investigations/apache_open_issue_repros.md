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

- Runtime/static-verified executable repros: 53 total: 47 open issues and 6 open PRs.
- Issue-side source/static-only candidates still pending runtime confirmation: 2 (`#19655`, `#20376`).
- PR-side source/static-only candidates still pending runtime confirmation: 2 (`#20470`, `#20316`).
- Broad-scan PR candidates not yet verified and not counted as current bugs: 18.
- Latest verification:
  `/home/smola/d2l-neu/.venv-mxnet/bin/python -m pytest -q tests/python/unittest/test_apache_open_issue_repros.py`
  reported `53 xfailed`; running the same file with `--runxfail` reported
  `53 failed`, confirming the repros detect current behavior.

## Runtime/API-Verified Repros

| GitHub item | Test | Current symptom |
|---|---|---|
| PR #21217 | `test_pr_21217_horovod_kvstore_exposes_barrier` | Horovod KVStore exposes no `_barrier` method. |
| issue #21176 | `test_issue_21176_conv2d_nhwc_cpu_runs` | CPU NHWC `Conv2D` reaches oneDNN and fails primitive creation. |
| PR #21044 | `test_pr_21044_symbolblock_preserves_symbol_parameter_attrs` | `SymbolBlock` drops user-provided `lr_mult`, `wd_mult`, and initializer attributes from symbols. |
| issue #21119 | `test_issue_21119_cross_gpu_binary_op_does_not_hang` | Cross-GPU NumPy binary op hangs instead of copying or rejecting contexts. |
| issue #21111 | `test_issue_21111_cudnn_batchnorm_cachedop_forward_only_train_mode_is_stateless` | cuDNN `BatchNorm` `CachedOp` mutates moving variance to NaN during forward-only train-mode execution. |
| issue #21156 | `test_issue_21156_indexed_recordio_close_survives_module_teardown` | `MXIndexedRecordIO.close()` depends on `MXIndexedRecordIO` module global during teardown. |
| issue #21146 | `test_issue_21146_gru_deferred_init_with_sequence_length_runs` | GRU with `use_sequence_length=True` fails in deferred-init/argument binding. |
| issue #20936 | `test_issue_20936_wheel_exposes_include_path` | Installed wheel has no include path usable by `mx.libinfo.find_include_path()`. |
| issue #20657 | `test_issue_20657_find_conf_path_env_override_is_sequence` | `find_conf_path()` returns a bare string for `MXNET_CONF_PATH`; callers index the first character. |
| issue #20605 | `test_issue_20605_csr_gradient_preserves_sparse_pattern` | CSR gradients keep CSR storage but densify to all entries instead of preserving the source sparse pattern. |
| issue #20577 | `test_issue_20577_symbolblock_export_succeeds_without_cached_op_args` | `SymbolBlock.export()` raises a raw missing `_cached_op_args` `AttributeError`. |
| issue #20391 | `test_issue_20391_numpy_gluon_allows_row_sparse_gradients` | NumPy/Gluon 2.0 rejects row-sparse gradients. |
| PR #20491 | `test_pr_20491_cpp_symbol_exposes_optimize_for_backend` | The C++ `Symbol` API does not expose `OptimizeForBackend`. |
| issue #20037 | `test_issue_20037_recordio_preserves_large_integer_label` | Scalar RecordIO labels round through float32. |
| issue #20180 | `test_issue_20180_box_encode_zero_refs_is_validated_or_empty` | `contrib.box_encode` with zero reference boxes fails through an internal `TBlob` shape error. |
| issue #20076 | `test_issue_20076_sequence_mask_rejects_huge_lengths_cleanly` | `SequenceMask` accepts huge `sequence_length` values and crashes instead of validating. |
| issue #20046 | `test_issue_20046_image_resize_invalid_interp_has_mxnet_validation` | `image.resize` forwards invalid interpolation ids into OpenCV instead of rejecting them in MXNet. |
| issue #20044 | `test_issue_20044_boolean_mask_empty_out_is_safe` | `contrib.boolean_mask` with empty data and `out=` crashes asynchronously. |
| issue #19860 | `test_issue_19860_swish_negative_beta_zero_input_is_finite` | `Swish(beta=-1e307)` returns NaN for zero input. |
| issue #19852 | `test_issue_19852_instancenorm_large_finite_input_is_finite` | `InstanceNorm` overflows large finite inputs into NaN outputs. |
| issue #19785 | `test_issue_19785_groupnorm_zero_groups_is_python_error_not_abort` | `GroupNorm(num_groups=0)` aborts with SIGFPE instead of a Python error. |
| issue #19753 | `test_issue_19753_topk_indices_are_integer_typed` | `topk(..., ret_typ='both')` returns float32 indices. |
| issue #19628 | `test_issue_19628_gpu_ctcloss_accepts_fp16_predictions` | GPU `CTCLoss` with FP16 predictions fails with an internal half-vs-float `TBlob` mismatch. |
| issue #19659 | `test_issue_19659_hybrid_boolean_mask_backward_runs` | Hybridized `boolean_mask` backward fails because required backward inputs are missing. |
| issue #19686 | `test_issue_19686_selfatt_qk_rejects_zero_heads_cleanly` | `interleaved_matmul_selfatt_qk(heads=0)` divides by zero or raises an unhelpful floating-point error. |
| issue #19683 | `test_issue_19683_arange_like_repeat_zero_is_safe` | `contrib.arange_like(repeat=0)` aborts instead of producing or validating an empty result. |
| issue #19647 | `test_issue_19647_optimize_for_missing_backend_raises` | `Symbol.optimize_for()` logs a missing backend error but still returns a symbol. |
| issue #19423 | `test_issue_19423_choice_full_without_replacement_is_permutation` | `np.random.choice(n, size=n, replace=False)` returns the identity range. |
| issue #19458 | `test_issue_19458_tensordot_scalar_empty_axes_backward` | `mx.np.tensordot` backward with scalar input and explicit empty axes fails with an internal shape mismatch. |
| issue #19422 | `test_issue_19422_numpy_array_iteration_yields_python_scalars` | Iterating an MXNet NumPy ndarray yields zero-dimensional MXNet arrays instead of Python scalars. |
| issue #19170 | `test_issue_19170_stepped_slice_shares_storage` | Stepped NumPy slicing returns a copy instead of a view. |
| PR #18583 | `test_pr_18583_cpp_symbol_exposes_partial_shape_inference` | The C++ `Symbol` API does not expose partial shape inference. |
| issue #19021 | `test_issue_19021_backward_rejects_mismatched_head_gradient_shape` | `backward()` accepts head gradients with shapes incompatible with the output and silently computes partial gradients. |
| issue #18919 | `test_issue_18919_numpy_advanced_indexing_matches_numpy` | NumPy advanced indexing rejects broadcast-compatible mixed index arrays. |
| issue #18770 | `test_issue_18770_non_native_byte_order_is_not_silently_lost` | Non-native NumPy byte order is silently discarded instead of preserved or rejected. |
| PR #18792 | `test_pr_18792_sort_and_argsort_support_float16` | `sort` and `argsort` reject float16 tensors. |
| issue #18669 | `test_issue_18669_zoneout_output_matches_new_state` | `ZoneoutCell` returns an output inconsistent with its first recurrent state. |
| issue #18563 | `test_issue_18563_max_backward_splits_tied_gradient` | `max` backward gives full gradient to every tied maximum. |
| issue #18078 | `test_issue_18078_prod_backward_multiple_zeros_is_finite` | `prod` backward with multiple zeros returns NaN gradients. |
| issue #18300 | `test_issue_18300_numpy_prod_accepts_shape_tuple` | `mxnet.numpy.prod` rejects shape tuples such as `array.shape`. |
| PR #17209 | `test_pr_17209_parameter_symbol_var_omits_dtype_attribute` | Gluon `Parameter.var()` still emits a fixed `__dtype__` attribute. |
| issue #17936 | `test_issue_17936_gammaln_promotes_integer_input` | `npx.gammaln(int32)` returns integer zeros. |
| issue #17698 | `test_issue_17698_split_and_load_does_not_materialize_full_input_first` | `split_and_load` first materializes the whole NumPy input on `ctx_list[0]`. |
| issue #11774 | `test_issue_11774_batchnorm_without_scale_or_center_trains` | `BatchNorm(scale=False, center=False)` loses the autograd graph during training-style backward. |
| issue #16402 | `test_issue_16402_legacy_ndarray_dtype_is_numpy_dtype_object` | Legacy `NDArray.dtype` is a scalar class, so `dtype()` returns `0.0`. |
| issue #16427 | `test_issue_16427_recordio_pack_accepts_python3_string_payload` | `recordio.pack()` concatenates Python 3 `str` payloads with bytes. |
| issue #13953 | `test_issue_13953_upsampling_accepts_data_keyword` | The symbolic `UpSampling` wrapper rejects the documented `data=` keyword. |
| issue #13945 | `test_issue_13945_indexed_recordio_shared_reader_is_thread_safe` | Concurrent `MXIndexedRecordIO.read_idx()` calls on one reader return records for the wrong key. |
| issue #13193 | `test_issue_13193_sparse_elemwise_mul_has_canonical_csr_payload` | Sparse CSR `elemwise_mul` has correct dense values but non-canonical, overallocated CSR payload metadata. |
| issue #8430 | `test_issue_8430_ndarrayiter_preserves_integer_label_dtype` | `NDArrayIter` converts large integer labels to float32 and loses precision. |
| issue #12286 | `test_issue_12286_ndarray_wrapper_raises_python_typeerror_for_missing_inputs` | Generated NDArray wrappers allow missing required inputs to reach backend `MXNetError` instead of Python `TypeError`. |
| issue #8817 | `test_issue_8817_sparse_zeros_accepts_integer_shape` | `mx.nd.sparse.zeros(..., shape=10)` rejects one-dimensional integer shapes. |
| issue #14695 | `test_issue_14695_single_output_ndarray_is_not_tuple_unpackable` | A single-output legacy `NDArray` result can be unpacked into multiple values. |

## Source-Verified Pending Runtime Repro

These were source-verified in the scan but still need an executable repro before
being promoted to the table above.

Issue-side source/static-only candidates still pending a runtime repro:
`#19655` and `#20376`.

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
`#19659`, `#19647`, `#19628`, `#19458`, `#19423`, `#19422`, `#19170`,
`#19021`, `#18919`, `#18770`, `#18669`, `#18563`, `#18300`, `#18078`,
`#17936`, `#17698`, `#16427`, `#16402`, `#13953`, `#13945`, `#13193`,
`#12286`, `#11774`, `#8817`, `#8430`, `#14695`.

### Issues: Source-Verified Only

`#19655`, `#20376`.

Notes:

- `#19655` needs a custom `optimize_for` backend to exercise the runtime path.
  Source still sets graph attrs and calls `PrePartition(...)` without first
  waiting on the supplied NDArrays.
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
`#18643`, `#18617`, `#18584`, `#18575`, `#18476`, `#18466`, `#18265`,
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
