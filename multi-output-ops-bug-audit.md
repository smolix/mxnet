# Multi-output op bug audit

Status: fixed for the confirmed `np.cross` issues.

This follows the `_npi_tril_indices` partial-output bug. The cached-op executor
correctly passes independent reqs such as `[kWriteTo, kNullOp]` for partially
consumed multi-output nodes; operators must not accidentally skip requested
outputs when a sibling req is `kNullOp`.

## Scan result

The exact `_npi_tril_indices` pattern, a shared kernel launch nested inside
`MXNET_ASSIGN_REQ_SWITCH` for multiple output reqs, does not appear broadly.
Most `req[1]` / `req[2]` uses are independent per-output assignments and are
safe to skip independently.

## Confirmed bugs

### 1. `np.cross` backward skips `grad_b` for 3D x 2D inputs

File: `src/operator/numpy/np_cross-inl.h`, in
`NumpyCrossBackwardImpl<xpu, DType, 3, 2>`, no-broadcast branch.

The code copies into `grad_b` under `MXNET_ASSIGN_REQ_SWITCH(req[0], ...)`
instead of `req[1]`:

```cpp
// Copy w1_data to grad_b.
MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
  mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
      s, grad_b.Size(), res_ptr, grad_b.dptr<DType>());
});
```

When only the second input requires gradients, `req[0] == kNullOp` and
`req[1] == kWriteTo`, so `grad_b` is left zero/unwritten.

Repro:

```python
import numpy as onp
import mxnet as mx
from mxnet import np, npx, autograd
npx.set_np()

a = np.array([1., 2., 3.], dtype="float32")
b = np.array([4., 5.], dtype="float32")
a.attach_grad("null")
b.attach_grad("write")

with autograd.record():
    loss = np.cross(a, b).sum()
loss.backward()

print(b.grad.asnumpy())  # current: [0. 0.], expected approximately [1. -2.]
```

Fix: guard the `grad_b` copy with `req[1]`, not `req[0]`.

### 2. `np.cross` backward misses broadcast reduction for lower-rank `b`

File: `src/operator/numpy/np_cross-inl.h`, `GetReduceAxis`.

`GetReduceAxis(move_shape, broad_move_shape)` only handles equal rank or
`move_shape.ndim() == broad_move_shape.ndim() + 1`. For normal broadcasting
from a lower-rank input to a higher-rank output, the relationship is the
opposite: `move_shape.ndim() + 1 == broad_move_shape.ndim()`.

This makes the backward path treat broadcasted `b` as if no reduction were
needed. The result is one slice of the gradient instead of the sum across the
broadcast axis.

Repro:

```python
import numpy as onp
import mxnet as mx
from mxnet import np, npx, autograd
npx.set_np()

a = np.array([[1., 2., 3.], [2., 3., 4.]], dtype="float32")
b = np.array([4., 5.], dtype="float32")
a.attach_grad("null")
b.attach_grad("write")

with autograd.record():
    loss = np.cross(a, b).sum()
loss.backward()

print(b.grad.asnumpy())  # current: [1. -2.], expected approximately [2. -4.]
```

If both inputs request gradients in this same broadcast case, the backward pass
can also fail with:

```text
no specialized NumpyCrossOp defined for template parameters
```

That appears to be the same shape/reduction-family bug rather than a separate
req-switch bug.

Fixes:

- Make `GetReduceAxis` use right-aligned broadcast semantics and distinguish
  scalar 2D-cross outputs from vector outputs.
- Reduce into a temporary shape with broadcast axes kept at size 1 before
  copying into the original gradient output.
- Map reduce axes back into original-axis coordinates when reduction happens
  after moving the vector axis away from the last dimension.
- Use typed workspace pointers when reshaping temporary `TBlob`s; passing
  `dptr_` selected the `void*` constructor and interpreted the CPU device id as
  a float32 type flag.

### 3. `np.cross` forward uses the wrong rank for broadcasted vector outputs

File: `src/operator/numpy/np_cross-inl.h`, generic
`NumpyCrossForwardImpl<xpu, DType, a_dim, b_dim>`.

The implementation derived `c_ndim` from `b_moveaxis_shape.ndim()` instead of
`c_moveaxis_shape.ndim()`. Lower-rank broadcast cases such as shape `(2, 3)`
cross `(2,)` produce an output with rank greater than `b`, and the generic
forward path then failed its vector-output shape check.

Fix: derive `c_ndim` from `c_moveaxis_shape.ndim()`.

### 4. `np.cross` backward type dispatch should follow input dtype

File: `src/operator/numpy/np_cross-inl.h`, `NumpyCrossBackward`.

The backward dispatcher selected `DType` from `grad_c.type_flag_`. The gradient
outputs and saved inputs are the values whose buffers are read/written by the
cross kernels, so the dispatch now follows `a.type_flag_`. In the reproduced
cases all type flags matched, but using the input dtype is the correct invariant
for the backward implementation.

## Checked and currently not implicated

- `split` symbolic partial binds: all requested outputs matched full-group bind.
- `topk(..., ret_typ="both")` symbolic partial binds: value-only and index-only
  outputs matched full-group bind.
- LAPACK-backed `qr` / `eig` / `svd` partial-bind runtime checks could not run
  in the current build because this local `libmxnet.so` was built without
  LAPACK.

## Regression tests

- `tests/python/unittest/test_numpy_op.py::test_np_cross_backward_second_input_only_req`
- `tests/python/unittest/test_numpy_op.py::test_np_cross_backward_lower_rank_broadcast_grad`
- Existing randomized `tests/python/unittest/test_numpy_op.py::test_np_cross`
  now also covers the corrected right-aligned reduction oracle.
