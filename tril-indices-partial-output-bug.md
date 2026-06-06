# Bug: single-output symbolic bind of a multi-output op leaves the consumed output unwritten

**Status:** root-caused; fix is in `src/operator/numpy/np_matrix_op-inl.h`.
**Severity:** correctness; **no d2l impact** (d2l uses imperative `np` and full-group
binds, both correct). Affects symbolic `_simple_bind` of *one* output of a
multi-output op.
**First surfaced by:** `tests/python/unittest/test_numpy_op.py::test_np_tril_indices_partial_outputs`
(added in commit `716370508` "Fix partial output request edge cases").

---

## Symptom

Binding **a single output** of a 2-output op symbolically returns the requested
output **unwritten** (stale/garbage or zeros). The failing unit test:

```python
row, col = mx.sym.np.tril_indices(4, k=0, m=4)   # op has 2 outputs
exe = row._simple_bind(ctx=mx.cpu())             # bind ONLY output 0
exe.forward(is_train=False)
exe.outputs[0].asnumpy()                          # WRONG
# expected: [0 1 1 2 2 2 3 3 3 3]
# actual:   [0 40769 0 0 0 0 0 0 0 0]   (uninitialized memory)
```

## What works vs what fails (reproduced)

| Path | Result |
|------|--------|
| Imperative `mx.np.tril_indices(4,k=0,m=4)` | ✅ correct (`row`, `col` both right) |
| Symbolic, bind **both** via `mx.sym.Group([row, col])` | ✅ both correct |
| Symbolic, bind **only `row`** (output 0) | ❌ output 0 unwritten (garbage) |
| Symbolic, bind **only `col`** (output 1) | ❌ output 0 all zeros (unwritten) |

So the op math and the full-graph path are fine; the failure is specific to a
multi-output node with **only some outputs consumed** in a symbolic bind.

Minimal repro script:

```python
import mxnet as mx, numpy as onp
print("imperative:", *[a.asnumpy() for a in mx.np.tril_indices(4, k=0, m=4)])
row, col = mx.sym.np.tril_indices(4, k=0, m=4)
g = mx.sym.Group([row, col])._simple_bind(ctx=mx.cpu()); g.forward(is_train=False)
print("group out0:", g.outputs[0].asnumpy())          # correct
r = row._simple_bind(ctx=mx.cpu()); r.forward(is_train=False)
print("row-only out0:", r.outputs[0].asnumpy())        # GARBAGE  <-- bug
```

## What is NOT the cause (ruled out by inspection)

- **The op kernel** — `TrilindicesOpForward` / `TrilindicesOpForwardImpl`
  (`src/operator/numpy/np_matrix_op-inl.h`, ~lines 408–484). `KERNEL_ASSIGN`
  honours `req` per output; `716370508` already split the single `req` into
  independent `req0`/`req1`, so the *unconsumed* output is correctly skipped.
  This is a real improvement but it does not explain why the **consumed**
  output is unwritten.
- **Shape inference** — `TrilindicesOpShape` (`src/operator/numpy/np_matrix_op.cc`,
  ~line 967) always assigns *both* outputs the shape `[length]`, so
  `out_data0.shape_[0]` is correct (10) and the `CHECK_EQ(out0.shape, out1.shape)`
  in the FCompute passes (it does not throw — we get garbage, not a CHECK abort).
- **cached_op `req` assignment (on inspection)** — in `src/imperative/cached_op.cc`
  the forward `array_reqs` are built (`~lines 775–782`) from a ref-count pass
  (`SetRefCounts`, `~lines 296–308`): graph-output entries get
  `ref_count >= 1 → kWriteTo`, unconsumed entries get `kNullOp`. For a single
  bound output this *should* leave the consumed output at `kWriteTo`. Reading the
  code it looks correct, so the defect is subtler than the obvious req mapping.

## Where the bug actually is

The symbolic/cached-op path is doing the right thing: for a single bound output
of `_npi_tril_indices`, it invokes the op with one output at `kWriteTo` and the
unused sibling at `kNullOp`.

The actual defect is the op's req-dispatch code in
`src/operator/numpy/np_matrix_op-inl.h`. `TrilindicesOpForward` used nested
`MXNET_ASSIGN_REQ_SWITCH(req[0], ...)` / `MXNET_ASSIGN_REQ_SWITCH(req[1], ...)`
around the single kernel launch. `MXNET_ASSIGN_REQ_SWITCH` intentionally skips
its body for `kNullOp`, so a partial-output invocation skipped the whole launch
whenever either sibling output was unused. Full-group symbolic bind and
imperative calls passed because both outputs were requested.

The correct pattern for a multi-output kernel whose launch can honor `kNullOp`
per output is `MXNET_REQ_TYPE_SWITCH` for each req, then `KERNEL_ASSIGN` inside
the kernel map. That switch preserves `kNullOp` as a compile-time req value
instead of suppressing the launch body.

Execution path for the repro:
`Symbol._simple_bind` (python/mxnet/symbol/symbol.py) →
`mxnet.executor.Executor` (python/mxnet/executor.py, which wraps
`ndarray.CachedOp(sym, flags=[("static_alloc", ...)])`) → C++
`src/imperative/cached_op.cc`. There is no separate legacy `src/executor/`
GraphExecutor in this 2.0 tree.

## Suggested investigation plan

1. Keep the runtime regression in
   `tests/python/unittest/test_numpy_op.py::test_np_tril_indices_partial_outputs`.
   It covers full-group bind plus single-output binds for both output slots.
2. Audit other multi-output operators for nested `MXNET_ASSIGN_REQ_SWITCH`
   around a shared launch. That pattern is only correct when the whole launch
   should be skipped if the switched req is `kNullOp`; it is wrong for kernels
   that can independently honor `kNullOp` per output.
3. If another multi-output partial-bind failure appears, first inspect req-switch
   structure before changing cached-op ref counts or memory planning.

## Effort & risk

- **Fix size:** small. The code change is the req-switch selection in
  `TrilindicesOpForward`, not cached-op memory planning.
- **Risk:** low-to-medium. The runtime behavior changes only for partial-output
  invocations where at least one req is `kNullOp`; full-output binds still launch
  the same kernel with write reqs for both outputs.

## Affected pattern

Any multi-output op that wraps a shared launch in nested
`MXNET_ASSIGN_REQ_SWITCH` blocks can have the same failure mode. Multi-output ops
that dispatch each output independently, or use `MXNET_REQ_TYPE_SWITCH` and
`KERNEL_ASSIGN`, are not implicated by this specific bug.
