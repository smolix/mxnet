# `mxnet.numpy.argmax` GPU bug — size-1 reduction axis returns row indices

Discovered: 2026-05-23
Wheel: `mxnet-2.0.0+cu13.bw.20260522.2-cp312-cp312-linux_x86_64`
Host: RTX 4090 (sm_89), NVIDIA driver 590.x / CUDA 13.x driver line
API surface: `mxnet.numpy` (the new NumPy-compatible front end), GPU only

## TL;DR

`np.argmax(x, axis=k)` returns `[0, 1, 2, ..., N-1]` instead of `[0, 0, ..., 0]`
when the `k`-axis has size 1 and the array is on a GPU context. CPU is fine.
`np.max` on the same input is fine. Reduction axes with size ≥ 2 are fine.

## Reproducer

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as _np
import mxnet as mx
from mxnet import np, npx
npx.set_np()

data = np.array(_np.random.rand(8, 1).astype('float32'), ctx=mx.gpu(0))

print("argmax(axis=1):", np.argmax(data, axis=1).asnumpy())
print("max(axis=1)   :", np.max(data, axis=1).asnumpy())
```

Expected output:

```
argmax(axis=1): [0 0 0 0 0 0 0 0]
max(axis=1)   : [<eight independent values, correct>]
```

Actual output on this wheel:

```
argmax(axis=1): [0 1 2 3 4 5 6 7]
max(axis=1)   : [<eight independent values, correct>]
```

## Scope

Verified by sweeping `axis_size ∈ {1, 2, 3, 5}`:

```python
for axis_size in (1, 2, 3, 5):
    data_np = _np.random.RandomState(0).rand(8, axis_size).astype('float32')
    data = np.array(data_np, ctx=mx.gpu(0))
    got = np.argmax(data, axis=1).asnumpy()
    expected = data_np.argmax(axis=1)
    print(f"axis_size={axis_size}  match={bool((got == expected).all())}  "
          f"got[:5]={got[:5].tolist()}  expected[:5]={expected[:5].tolist()}")
```

| `axis_size` | Device | Result |
|------------:|:------:|:-------|
| 1           | CPU    | correct (`[0, 0, …]`) |
| 1           | GPU    | **broken** — returns `[0, 1, 2, …, N-1]` |
| 2           | GPU    | correct |
| 3           | GPU    | correct |
| 5           | GPU    | correct |

`np.max(data, axis=1)` is correct in all cases, including `axis_size == 1` on
GPU. So the reduction itself works; only the index produced by `argmax` is
wrong, and only for the degenerate size-1 axis.

## Suspected location

The GPU branch of `mxnet.numpy.argmax`. Almost certainly the kernel registered
for `_npi_argmax` (or the underlying reduction kernel that backs it) in
`src/operator/numpy/np_broadcast_reduce_op_value.cu` (or wherever the
new-API argmax lives in this tree).

Likely cause: when the reduction axis has size 1 the per-output inner loop
never executes (zero-trip), and the accumulator's initial index ends up being
the *outer* coordinate instead of `0`. The CPU branch must take a different
code path (the loop body runs at least once even for size-1 axes, or the
accumulator is initialized to `0` regardless of trip count), which is why it
is correct.

The legacy `mx.nd.argmax` API was not tested here — worth checking whether it
shares the kernel and exhibits the same bug before patching.

## Downstream impact: d2l SSD failure

This was discovered while diagnosing `chapter_computer-vision/ssd.ipynb`
failing under the wheel with:

```
IndexError: index 5362 is out of bounds for axis 0 with size 1
```

The trail:

1. `d2l.assign_anchor_to_bbox` in `chapter_computer-vision/anchor.md` (mxnet
   branch) calls `indices = np.argmax(jaccard, axis=1)` where
   `jaccard.shape == (num_anchors, num_gt_boxes)`.
2. The banana detection dataset used by SSD has **exactly one** ground-truth
   box per image, so `num_gt_boxes == 1` and `jaccard.shape == (5444, 1)` on
   GPU. This is the exact bug case.
3. `indices` comes back as `[0, 1, 2, …, 5443]` instead of all zeros.
4. `box_j = indices[max_ious >= 0.5]` then contains anchor indices
   (e.g. `[277, 569, 688, …, 5362]`) instead of gt-box indices (all 0).
5. `anchors_bbox_map[anc_i] = box_j` correctly stores those wrong values.
6. `bb_idx = anchors_bbox_map[indices_true]` therefore contains values up to
   `num_anchors - 1`.
7. `label[bb_idx, ...]` finally throws, because `label.shape == (1, 5)` and
   `bb_idx` contains 5362. The IndexError surfaces in `multibox_target` but
   the corruption originates in step 3.

## Workaround on the d2l side

`chapter_computer-vision/anchor.md`, mxnet branch of `assign_anchor_to_bbox`,
now special-cases `num_gt_boxes == 1`:

```python
max_ious = np.max(jaccard, axis=1)
if num_gt_boxes == 1:
    # mxnet 2.0+cu13.bw.20260522.2 GPU bug: np.argmax(x, axis=k) returns row
    # indices instead of zeros when the k-axis has size 1. Sidestep here.
    indices = np.zeros((num_anchors,), dtype='int64', ctx=device)
else:
    indices = np.argmax(jaccard, axis=1)
```

Revert this once the kernel is fixed.

## Suggested verification after the kernel fix

In addition to the reproducer above, add unit tests for `np.argmax` along
each reduction axis at sizes `{1, 2, 3, 8}`, on both `mx.cpu()` and
`mx.gpu(0)`. The current size-1 hole strongly suggests the existing argmax
tests don't exercise that case.

Also verify whether `mx.nd.argmax(...)` (legacy API, not `mxnet.numpy`)
shares the same kernel and shows the same behavior — same fix may be needed
there.
