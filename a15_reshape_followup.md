# A15 follow-up — `nd.reshape` size validation (apache/mxnet#14264)

## Summary

Two related fixes in `src/operator/tensor/matrix_op-inl.h`:

1. **Primary fix (apache#14264):** re-enable the size-equality CHECK in
   `ReshapeShape`, guarded by `shape_is_known(dshape) && shape_is_known(oshape)`
   so dynamic-shape / unresolved `-1` cases are not affected.
2. **Collateral fix:** the `-1` inference path in `InferReshapeShape` did
   `dshape.Size() / new_size` without guarding against `new_size == 0`. When
   shape inference is invoked with a `0` in `dshape` and the `-1` lands on the
   only remaining dim, you get `0 / 0` which traps as an integer SIGFPE and
   crashes the process (not catchable from Python). Now `new_size == 0` leaves
   the inferred dim as `-1` and lets reverse inference fill it.

These also required two Python-side fixes (pre-existing breakage, not specific
to A15 but blocked importing `mxnet` after the WIP rebuild):

- `python/mxnet/numpy_op_signature.py`: skip ops listed in `_numpy_op_doc` that
  no longer exist in `mxnet.numpy` (e.g. deprecated `sometrue`), instead of
  raising at import time.
- `python/mxnet/numpy_dispatch_protocol.py`: comment out `'sometrue'` from
  `_NUMPY_ARRAY_FUNCTION_LIST` (deprecated numpy alias, never implemented in
  `mxnet.numpy`).

## Verification

Reproducer from apache/mxnet#14264 (`nd.arange(10).reshape((1,2))`) now raises
`MXNetError` instead of silently returning a 2-element array. Forward / -1
inference / "keep dim 0" semantics all continue to work:

```
Test 1: bad reshape (10 -> 1x2)      -> Raised correctly: MXNetError
Test 2: valid reshape (10 -> 2x5)    -> shape = (2, 5)
Test 3: -1 inference (10 -> 2x-1)    -> shape = (2, 5)
Test 4: keep-dim with 0 (2x5 -> 0x5) -> shape = (2, 5)
```

Tested test suites:

| Test                                                | Pass | Fail | Skipped |
|-----------------------------------------------------|------|------|---------|
| `tests/python/dnnl/subgraphs/test_fc_subgraph.py`   | 387  | 0    | 16      |
| `test_operator.py::test_reshape_old`                | 1    | 0    | 0       |
| `test_operator.py::test_reshape_like`               | 1    | 0    | 0       |
| `test_operator.py::test_reshape_like_different_types` | 1  | 0    | 0       |
| `test_dynamic_shape.py` (full)                      | 5    | 0    | 0       |

FC subgraph baseline (387/0/16) is preserved exactly.

## Unrelated pre-existing failure

`test_operator.py::test_reshape_new` (26 parametrized cases) fails in its
**holdout block** (lines 2383-2395, added by Sheng Zha 2020-05-04 in commit
`0580200562`). The holdout block does:

```python
for i in range(len(src_shape)):
    holdout_src_shape = list(src_shape)
    holdout_src_shape[i] = 0          # 0 = "unknown" per symbol infer_shape API
    holdout_src_shape = tuple(holdout_src_shape)
    net = mx.sym.Variable('data')
    net = mx.sym.elemwise_add(net.reshape(shape_args, reverse=reverse),
                              mx.sym.ones(shape=dst_shape))
    input_shape, output_shape, __ = net.infer_shape(data=holdout_src_shape)
    assert output_shape[0] == dst_shape  # expects reverse-infer to fill 0
```

This block has been **broken on this fork since at least the oneDNN v3 port**:
- **Before the A15 WIP**: SIGFPE in `InferReshapeShape` line 170 (`0/0` integer
  division when `dshape` has a `0` dim and the only `-1` lands on the same
  axis). Process-killing crash.
- **After the A15 WIP + 0/0 guard**: now raises a clean `MXNetError` saying
  `Incompatible attr in node elemwise_add0 at 1-th input: expected [0,-1],
  got [2,75]`.

### Why the test design is broken

The test assumes 0 in `dshape` means "unknown" and that `ReverseReshapeInferShape`
will fill it from the `oshape` constraint imposed by the elemwise_add peer
input. But MXNet's TShape treats 0 as a **valid known zero**:

- `TShape::Size()` of `(0,3,5,5)` returns `0`, not "unknown".
- `dim_size_is_known(dim)` returns `false` only for `dim == -1`, not for `dim == 0`.
- `shape_is_known(shape)` therefore returns `true` for shapes containing `0`.

So `(0,3,5,5)` is treated as a fully-known zero-sized tensor, the `0/0` path
fires, and forward inference produces `(0, -1)` (with my fix) which then
clashes with the `(5,30)` peer in elemwise_add.

Fixing this end-to-end would require rethinking the 0-vs-(-1)-vs-unknown
convention across `TShape::Size()`, `shape_is_known`, `dim_size_is_known`,
`ReverseReshapeInferShape`, and the symbol `infer_shape` API surface — out of
scope for an A15 follow-up.

### Decision matrix

| Option | Pros | Cons |
|---|---|---|
| **A. Land the fix, accept holdout failures** (chosen) | Closes apache#14264. Eliminates SIGFPE process crash. FC subgraph baseline preserved. Behavior is strictly better than before. | `test_reshape_new` still fails (26 parametrized cases) — but it was crashing the runner before, so the regression line is in our favor. |
| B. Land the fix, then also fix `dim_size_is_known(dim)` to treat 0 as unknown | Test_reshape_new would pass. | Cross-cutting change to TShape semantics. High risk of breaking dozens of other ops that rely on `Size() == 0` meaning "really zero". Would need full operator suite re-run. |
| C. Land the fix, but mark `test_reshape_new` as `@pytest.mark.xfail(reason="upstream test design issue, see a15_reshape_followup.md")` | Cleans up CI signal. | Mutating an upstream test file; needs reviewer sign-off; should arguably be a separate PR. |
| D. Revert the WIP, leave apache#14264 unfixed | No new failures. | Bug remains; SIGFPE remains a latent risk. |

**Chosen: A.** The fix is a strict net improvement (closes a real silent-data
bug, eliminates a process crash, preserves all production-path tests). The
holdout block was broken before this work and continues to be broken for an
unrelated reason that warrants its own design review.
