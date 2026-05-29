# Bug: single-output symbolic bind of a multi-output op leaves the consumed output unwritten

**Status:** open, deferred to a dedicated session.
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

In **`cached_op`'s execution / memory-planning of a multi-output node when only a
subset of its outputs is consumed**. The consumed output's buffer never receives
the kernel's write — i.e. at execution time its effective `req` is `kNullOp`, or
its storage is aliased to a buffer that is never the one exposed as
`exe.outputs[0]`, or the op is pruned/skipped. This is graph-executor
infrastructure shared by **every** multi-output op.

Execution path for the repro:
`Symbol._simple_bind` (python/mxnet/symbol/symbol.py) →
`mxnet.executor.Executor` (python/mxnet/executor.py, which wraps
`ndarray.CachedOp(sym, flags=[("static_alloc", ...)])`) → C++
`src/imperative/cached_op.cc`. There is no separate legacy `src/executor/`
GraphExecutor in this 2.0 tree.

## Suggested investigation plan

1. **Instrument** the cached_op forward loop and `TrilindicesOpForward` to print,
   for both the full-group bind and the single-output bind:
   - the `req` vector actually passed to FCompute,
   - `outputs[i].dptr_` and `outputs[i].shape_` for i in {0,1},
   - whether FCompute is even invoked for the node.
   Compare the two binds; the divergence pinpoints the layer.
2. **Reduce** to a minimal custom 2-output op (no temp-space, trivial kernel) to
   confirm the bug is generic to multi-output partial bind, not tril-specific.
3. **Inspect the memory plan**: how storage and `ref_count`/`kNullOp` are assigned
   to the consumed vs unconsumed output entries when the sibling output is unused.
   Likely-fruitful hypothesis: the consumed output entry's `req` is forced to
   `kNullOp` (or its storage id collides/aliases) because the node's *other*
   output being unused mis-marks the whole node's outputs.
4. Once root-caused, fix in `cached_op.cc` (likely a few lines in req/storage
   assignment), then **verify against every multi-output op** — `split`,
   `topk`/`sort`, `SVD`/`qr`/`eig` (`linalg`), `unique`, `tril_indices`/
   `triu_indices`, RNN state outputs — under single-output symbolic bind, plus the
   `716370508` regression tests.

## Effort & risk

- **~half a day.** The fix itself is probably small (a handful of lines in the
  req/storage assignment), but root-causing needs an instrument → rebuild → read
  loop, and **each rebuild is a full ~30-min recompile on 64 cores** because the
  embedded git commit hash lives in a widely-included header, so any commit
  invalidates almost every object. For debug iterations, consider pinning a dummy
  commit hash (or instrumenting via a `.cc` rather than a header) to keep the
  recompile scope small.
- **Risk: medium-high.** The fix touches core graph-executor infrastructure shared
  by all multi-output ops; a wrong change can silently corrupt other ops. Gate it
  behind the broad multi-output test sweep above.

## Affected ops (single-output symbolic bind)

Any op with `set_num_outputs > 1`: `split`, `topk`/`sort` (when both value+index
are defined but one is bound), `linalg` decompositions (`qr`, `svd`, `eig`,
`slogdet`), `unique`, `tril_indices`/`triu_indices`, RNN (state outputs), etc.
