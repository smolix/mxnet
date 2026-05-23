# D2L-NEU issues against `mxnet 2.0.0+cu13.bw.20260522.2`

Discovered: 2026-05-23
Wheel: `mxnet-2.0.0+cu13.bw.20260522.2-cp312-cp312-linux_x86_64`
Host: 4× NVIDIA RTX 4090 (sm_89, 24 GB), NVIDIA driver 590.x / CUDA 13.x driver line
API surface: `mxnet.numpy` (the new NumPy-compatible front end)

Test driver: `make run-notebooks-mxnet` from `~/d2l-neu` with
`NUM_GPUS=4 GPU_SLOTS=8 CPU_SLOTS=4`. 128 notebooks scheduled. Outcome:
**~124 / 128 pass**; the four remaining failures correspond to the
issues below.

The issues are ordered by severity from a d2l-neu perspective (issue 1
blocks SSD entirely; the others are scattered or quality-of-life).

---

## Issue 1: `mxnet.numpy.argmax` GPU bug — size-1 reduction axis returns row indices (BLOCKING for SSD)

`np.argmax(x, axis=k)` returns `[0, 1, 2, ..., N-1]` instead of
`[0, 0, ..., 0]` when the `k`-axis has size 1 and the array is on a GPU
context. CPU is fine. `np.max` on the same input is fine. Reduction axes
with size ≥ 2 are fine.

### Reproducer

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

Expected:

```
argmax(axis=1): [0 0 0 0 0 0 0 0]
max(axis=1)   : [<eight independent values, correct>]
```

Actual:

```
argmax(axis=1): [0 1 2 3 4 5 6 7]
max(axis=1)   : [<eight independent values, correct>]
```

### Scope

Verified by sweeping `axis_size ∈ {1, 2, 3, 5}`:

| `axis_size` | Device | Result |
|------------:|:------:|:-------|
| 1           | CPU    | correct (`[0, 0, …]`) |
| 1           | GPU    | **broken** — returns `[0, 1, 2, …, N-1]` |
| 2           | GPU    | correct |
| 3           | GPU    | correct |
| 5           | GPU    | correct |

`np.max(data, axis=1)` is correct in all cases, including `axis_size == 1`
on GPU. So the reduction itself works; only the index produced by `argmax`
is wrong, and only for the degenerate size-1 axis.

### Suspected location

The GPU branch of `mxnet.numpy.argmax`. Almost certainly the kernel
registered for `_npi_argmax` (or the underlying reduction kernel that
backs it) in `src/operator/numpy/np_broadcast_reduce_op_value.cu` (or
wherever the new-API argmax lives in this tree).

Likely cause: when the reduction axis has size 1 the per-output inner loop
never executes (zero-trip), and the accumulator's initial index ends up
being the *outer* coordinate instead of `0`. The CPU branch must take a
different code path (the loop body runs at least once even for size-1
axes, or the accumulator is initialized to `0` regardless of trip count),
which is why it is correct.

The legacy `mx.nd.argmax` API was not tested here — worth checking whether
it shares the kernel and exhibits the same bug before patching.

### Downstream impact: d2l SSD failure

`chapter_computer-vision/ssd.ipynb` fails under the wheel with:

```
IndexError: index 5362 is out of bounds for axis 0 with size 1
```

The trail:

1. `d2l.assign_anchor_to_bbox` (mxnet branch) calls
   `indices = np.argmax(jaccard, axis=1)` where
   `jaccard.shape == (num_anchors, num_gt_boxes)`.
2. The banana detection dataset has **exactly one** ground-truth box per
   image, so `num_gt_boxes == 1` and `jaccard.shape == (5444, 1)` on GPU.
   This is the exact bug case.
3. `indices` comes back as `[0, 1, 2, …, 5443]` instead of all zeros.
4. `box_j = indices[max_ious >= 0.5]` then contains anchor indices
   (e.g. `[277, 569, 688, …, 5362]`) instead of gt-box indices (all 0).
5. `anchors_bbox_map[anc_i] = box_j` correctly stores those wrong values.
6. `bb_idx = anchors_bbox_map[indices_true]` therefore contains values up
   to `num_anchors - 1`.
7. `label[bb_idx, ...]` finally throws, because `label.shape == (1, 5)`
   and `bb_idx` contains 5362. The IndexError surfaces in
   `multibox_target` but the corruption originates in step 3.

### Suggested verification after the kernel fix

Add unit tests for `np.argmax` along each reduction axis at sizes
`{1, 2, 3, 8}`, on both `mx.cpu()` and `mx.gpu(0)`. The current size-1
hole strongly suggests the existing argmax tests don't exercise that case.

Also verify whether `mx.nd.argmax(...)` (legacy API, not `mxnet.numpy`)
shares the same kernel and shows the same behaviour — same fix may be
needed there.

---

## Issue 2: GPU OOM when two large models share a 24 GB GPU

Under the d2l-neu scheduler running with `GPU_SLOTS=8` on 4× 24 GB GPUs
(i.e. two notebook processes per GPU), two specific notebooks failed with:

```
MXNetError: Memory allocation failed out of memory
```

| Notebook | Wall time before OOM | Neighbour on same GPU at moment of OOM |
|---|---|---|
| `chapter_convolutional-modern/cnn-design.ipynb` | 685 s | `bert-pretraining.ipynb` |
| `chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb` | 83 s | `bert-pretraining.ipynb` |

Both failures occurred while a BERT-pretraining process was running on the
same physical GPU. Single-stream reruns of either notebook (no neighbour)
pass.

Two things make this worth a look from the mxnet side rather than chalking
it up to "the user asked for too much":

1. PyTorch, JAX, and TensorFlow execute the **same** notebooks on the
   **same** 4090 hardware under the **same** two-slots-per-GPU scheduler
   without OOM. So 24 GB is enough headroom for these workloads when the
   framework manages memory tightly.
2. The CNN-Design and Sentiment-RNN models are not large compared to BERT
   — yet BERT-pretraining survives the contention and the smaller model
   is the one that OOMs. That pattern is what you'd see if the smaller
   process is slower to release intermediate allocations between minibatches.

Cheap things to check: storage-pool fragmentation behaviour, idle-allocation
release between minibatches, and whether the pooled allocator returns to
the device when an arena hasn't been used for N seconds.

Repro hint: run `cnn-design.ipynb` and `bert-pretraining.ipynb` concurrently
with `CUDA_VISIBLE_DEVICES=0` set on both. Expect at least one to OOM.

---

## Issue 3: Dead kernel in `natural-language-inference-bert.ipynb`

```
nbclient.exceptions.DeadKernelError: Kernel died
```

The kernel died **1095 seconds (~18 minutes) into execution** — so deep into
training, not at startup or first allocation. There is no Python traceback;
the worker process disappeared without raising.

This pattern (long-running, then sudden death, no traceback) is what you
get from a native segfault or OS-side OOM kill. It is also one of the
failure signatures called out in `d2l-neu/docs/mxnet-runtime-diagnostics.md`
("Dead Kernels" section from the .20260519 wheel run).

Reproducer: run `chapter_natural-language-processing-applications/natural-language-inference-bert.ipynb`
from `_notebooks/mxnet/` under `.venv-mxnet/bin/jupyter nbconvert --execute
--inplace`. If it segfaults, `dmesg | tail` and a core dump (`ulimit -c
unlimited`) should pinpoint the offending kernel.

This may or may not be related to Issue 2 (the kernel could also have been
OOM-killed). Capturing a core dump or running under `cuda-gdb` would tell.

---

## Issue 4: `mxnet.__version__` reports the wrong (stale) version

The installed wheel's `dist-info/METADATA` correctly says:

```
Version: 2.0.0+cu13.bw.20260522.2
```

But at runtime:

```python
>>> import mxnet
>>> mxnet.__version__
'2.0.0+cu13.bw.20260518.1'
```

So the version string is baked into the build at an earlier point than
the wheel-naming step picks up. Not a runtime correctness issue, but it
makes the in-Python version string useless for telling apart `.20260518.1`,
`.20260518.2`, `.20260522`, `.20260522.1`, and `.20260522.2` — every
post-`.20260518.1` build I've installed reports the same string. This
matters in `d2l-neu` because notebook outputs frequently print
`mxnet.__version__` for provenance.

Likely fix: regenerate the version-string header (`include/mxnet/base.h` or
wherever `MXNET_VERSION` lives) from the same source as the wheel-naming
script, on every build.

---

## Issue 5: Noisy "Using Pooled (Naive) StorageManager" log line

On the **first allocation** in each device context, a banner appears on
stderr/stdout:

```
[02:51:45] /home/smola/mxnet/src/storage/storage.cc:202: Using Pooled (Naive) StorageManager for GPU
[02:51:45] /home/smola/mxnet/src/storage/storage.cc:202: Using Pooled (Naive) StorageManager for CPU
```

This shows up in every executed d2l-neu notebook's first output cell. It
isn't an error, but:

- It pollutes the rendered HTML book and PDFs with internal-source path
  references.
- It defeats `tools/inject_outputs.py` deduplication when the timestamp
  is non-deterministic.

Suggested fix: gate the message behind `MXNET_LOG_VERBOSITY` (or an
equivalent flag) and leave it off by default. Print at most once per
process, not per device.

---

## Things that USED to fail and now pass

Documented here so the mxnet team has a regression-test target if any of
them regress again:

| Notebook | Symptom under earlier wheel | Wheel where it was last seen broken |
|---|---|---|
| `chapter_convolutional-neural-networks/channels.ipynb` | `AssertionError` on CPU | `.20260518.2` |
| ~120 other GPU notebooks | `cudaErrorNoKernelImageForDevice` | `.20260518.2` |

`channels` is the most surprising one — it ran on **CPU** and still
asserted under `.20260518.2`. I don't have a reproducer for the CPU
regression, since the wheel that exhibited it is no longer installed; if
the mxnet team is interested, the failed `.ipynb` outputs from that run
are at `~/d2l-neu/logs/run-mxnet-20260523-015447.log` and
`~/d2l-neu/logs/run-mxnet-20260522-234224.log` on this host.
