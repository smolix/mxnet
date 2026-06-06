# NumPy-vs-PyTorch convention sweep (branch fix-bf16-dnnl-unittest-bugs)

Goal: where the branch follows NumPy but PyTorch deviates, follow PyTorch (plus
project overrides). Verified empirically with torch 2.12 (CPU) + numpy 1.26.4.

## Decisions applied
- int x float binary promotion: keep the FLOAT operand's width (PyTorch), not
  NumPy's float64. e.g. int32 x float16 -> float16.
- result_type: float-width for arrays; Python scalars are "weak" (category only)
  -> int32 array + python float -> float32.
- var/std/norm/average over integer input -> float32.
- mean over integer input -> int32 (with rounding)  [project override]
- accept int matmul/tensordot/cross and unique(float16)  [PyTorch parity]

## Status of changes
| Item | NumPy | PyTorch | Decision | Where | Done? |
|------|-------|---------|----------|-------|-------|
| int×float binary promote | float64 | float-width | float-width | src/common/utils.h type_promotion | YES (rebuilding) |
| result_type arrays | float64 | float-width | float-width | python .../numpy/utils.py table | YES (python) |
| result_type weak scalars | float64 | float32 | float32 | python .../numpy/type_functions.py | YES (python) |
| linalg.norm(int) | float64 | error | float32 | np_norm.cc NumpyNormType | YES (rebuilding) |
| var/std(int) | float64 | error | float32 | np_moments_op.cc (already GetDefaultDtype=f32) | already OK |
| average(int) | float64 | n/a | float32 | python average wrapper | verify post-rebuild |
| mean(int) | float64 | error | int32+round | np_broadcast_reduce_op_value.h NumpyMeanType + kernel | TODO (needs rounding kernel) |
| matmul/tensordot/dot(int) | int | int | accept | kernels are MSHADOW_REAL_TYPE_SWITCH (float-only, BLAS) | BLOCKED: needs integer GEMM kernel |
| cross(int) | int | int | accept | np_cross-inl.h MSHADOW_SGL_DBL_TYPE_SWITCH + check | TODO (kernel type-switch + grad) |
| unique(float16) | f16 | f16 | accept | np_unique_op.cc macro LOG(FATAL) on f16 | TODO (add f16 case to sort macro) |

## Branch behaviors that ALREADY match PyTorch (keep as-is)
- linalg qr/eig/eigh/svd/cholesky reject bool & float16 -> torch.linalg requires float32+/complex.
- unique(bool), sort(bool), topk(bool) accepted -> torch allows.
- take/index: reject float indices, bounds-check, negative-index wrap -> torch matches.
  (MXNet also rejects int16 indices at compute time, matching torch's long/int/bool-only rule.)

## Verification (full suite, 12,474 tests, GPU build)
- Baseline (NumPy-convention branch): 42 failures.
- After promotion/result_type/norm fixes: 41 failures, **0 new failures introduced** by the
  central type_promotion change; the mixed_precision_binary_funcs backward crash is FIXED.
- After migrating 39 legacy take/ravel/unravel tests to integer indices: 2 failures remain,
  both environmental (test_np_rand, test_np_randint -> scipy 1.13 chisquare strictness).

## Kernel-level items — final status
- unique(float16): DONE. Enabled the float16 case in MXNET_UNIQUE_TYPE_SWITCH_WITH_BOOL
  (CPU .cc + GPU .cu) and dropped the type-check; test flipped to test_np_unique_float16.
- cross(int): DONE. Widened the forward type switch to MSHADOW_TYPE_SWITCH (int8/uint8/
  int32/int64 + float16/32/64), made a sign-flip ternary type-stable, and set the type
  check to that supported set. Tests: test_np_cross_integer_supported (new) +
  test_np_cross_unsupported_dtypes_rejected (bool/int16/uint16/uint32 still rejected).
- mean(int)->int32+round: NOT a safe drop-in. The reduce (ReduceAxesComputeImpl) grabs the
  kTempSpace resource for its own workspace, so a float intermediate from that pool collides;
  heap NDArray temps would be freed at FCompute return while async GPU kernels still run
  (use-after-free). Needs a fused integer-mean-with-rounding reduce (real kernel work).
- matmul/tensordot/dot(int): NOT a safe drop-in for the same reasons + there is no integer
  GEMM (kernels are MSHADOW_REAL_TYPE_SWITCH over BLAS). A float64 cast-fallback hits the
  same temp-collision / async-lifetime hazard. Needs an integer GEMM kernel or an
  engine-var-tracked scratch. Left rejected.

## Feasibility notes
- int matmul/tensordot/dot: MXNet uses BLAS (float GEMM). No integer GEMM path exists;
  removing the dtype CHECK alone will NOT make it work. Requires a new fallback integer
  matmul kernel -> substantial. RECOMMEND: keep rejecting OR scope a dedicated task.
- cross(int): cross is elementwise (no BLAS); enabling needs widening the type switch and
  ensuring the backward handles int. Tractable but the branch's cross backward is complex.
- unique(float16): the kernel macro explicitly LOG(FATAL)s on float16; needs a float16
  case added to the sort path. Tractable.
