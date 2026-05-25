# D2L-NEU issues against `mxnet 2.0.0+cu13.bw.*`

Last updated: 2026-05-23 (evening pass)  
Current wheel: `mxnet-2.0.0+cu13.bw.20260523.6-cp312-cp312-linux_x86_64`  
Host: 4× NVIDIA RTX 4090 (sm_89 / Ada Lovelace, 24 GB each), NVIDIA driver 590.x / CUDA 13.x driver line  
API surface: `mxnet.numpy` (new NumPy-compatible front end) + `mxnet.gluon`

Test driver: `make run-notebooks-mxnet` from `~/d2l-neu` with
`NUM_GPUS=4 GPU_SLOTS=8 CPU_SLOTS=4`. 128 notebooks scheduled.  
**Outcome on .20260523.6 (initial 126/128 report): see per-issue status below.**

## Status snapshot (2026-05-23 PM follow-up)

| # | Issue | Status |
|---|---|---|
| 1 | `np.argmax` GPU size-1 axis | **Resolved** in .6 build; regression test landed in `tests/python/gpu/test_d2l_argmax_size_one_axis_regression.py` (19 tests, both `np.argmax` and `nd.argmax` covered) |
| 2 | GPU OOM under cross-process contention | **Mitigated** in .6 — retry-with-backoff path lives in `src/storage/pooled_storage_manager.h`; env: `MXNET_GPU_MEM_POOL_OOM_RETRIES` (default 4) / `MXNET_GPU_MEM_POOL_OOM_BACKOFF_MS` (default 50). Cross-process retry tests in `tests/python/gpu/test_d2l_bug_2_gpu_oom_retry.py` |
| 3 | `fine-tuning` DeadKernel | **Under investigation** — empirically NOT in autotune (also reproduces with `MXNET_CUDNN_AUTOTUNE_DEFAULT=0`); appears to be a multi-GPU dispatch instability in d2l's `train_ch13` driving 4× RTX 4090. Single-GPU runs intermittently pass; the synthetic two-net repro passes reliably. Defense-in-depth lock added in `src/operator/cudnn_ops.cc` `SelectPlan` against multi-thread cuDNN backend races; root cause still being narrowed |
| 4 | Stale version string | **Resolved** in .6 (build-time version regenerated against MXNET_PACKAGE_VERSION) |
| 5 | Storage banner noise | **Resolved** in .6 — gated behind `MXNET_LOG_STORAGE_INIT=1` in `src/storage/storage.cc:201-209`; regression test in `tests/python/unittest/test_d2l_storage_banner_suppression.py` |
| 6 | `lr-scheduler` convergence | **Root cause identified, framework-side fix shipped** — measurement confirmed MXNet/PyTorch/JAX schedulers are semantically equivalent (all count caller-supplied steps).  Added `epoch_size=` kwarg to `FactorScheduler`/`MultiFactorScheduler`/`PolyScheduler`/`CosineScheduler` so callers can drive epoch-rate decay under Gluon Trainer's per-minibatch counter (10 regression tests in `tests/python/unittest/test_d2l_lr_scheduler_epoch_size.py`). **d2l-side fix required**: rewrite the notebook to use `epoch_size=num_batches` |
| 7 | FCN convergence gap | **Root cause identified, NOT a loss-reduction bug** — `gluon.loss.SoftmaxCrossEntropyLoss(axis=1)`, `F.cross_entropy`, and `optax.softmax_cross_entropy_with_integer_labels` produce **bit-identical** mean loss (3.428971 on the synthetic input). The actual cause is the Gluon convention that `Trainer.step(batch_size)` rescales the gradient by `1/batch_size`, while PyTorch's `optimizer.step()` does not. For `lr=0.001` + `batch_size=32`, MXNet's effective LR is **32× smaller** than PyTorch's. `Trainer.step()` docstring updated; 3 regression tests in `tests/python/unittest/test_d2l_trainer_rescale_semantics.py`. **d2l-side fix required**: either call `trainer.step(1)` and adjust `lr`, or scale `lr` by `batch_size` |

The detail sections below preserve the original failure analyses for reference.

The issues are ordered by severity.

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

**Note (wheel .20260523.6):** The argmax bug is fixed in this wheel —
`chapter_computer-vision/ssd.ipynb` now passes (class_err=3.52e-3,
bbox_mae=3.76e-3), within ~10% of PyTorch and JAX. The fix should be
covered by a regression test (see below).

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
| `chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb` | 83–85 s | `bert-pretraining.ipynb` |

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

For the full `sentiment-analysis-rnn` isolation report, including the
exact OOM numbers (pool used 24 728 231 936 bytes / device free ~1 MB),
model code, and repro script, see:
`/home/smola/mxnet/d2l-issues/sentiment-analysis-rnn.md`.

---

## Issue 3: DeadKernel in `chapter_computer-vision/fine-tuning.ipynb` — cuDNN autotune crash on Ada

**Status as of wheel .20260523.6:** `fine-tuning` still fails (DeadKernel at 83 s). The
previously-noted `natural-language-inference-bert.ipynb` DeadKernel is **resolved** in this
wheel — NLI-BERT now passes. The remaining crash is isolated to the fine-tuning notebook.

The kernel dies silently — no Python exception, no MXNet error string — consistent with a
native-layer abort or SIGSEGV. The last lines on stderr before process exit are:

```
[07:20:41] /home/smola/mxnet/src/operator/cudnn_ops.cc:430: Auto-tuning cuDNN op, set MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable
[07:20:42] /home/smola/mxnet/src/operator/cudnn_ops.cc:430: Auto-tuning cuDNN op, set MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable
[NbConvertApp] ERROR | Kernel died while waiting for execute reply.
nbclient.exceptions.DeadKernelError: Kernel died
```

The crash occurs inside cuDNN autotuning during the first forward pass of a Gluon-hybridized
ResNet-18 on GPU (after ~80 s of model construction and pretrained-weight download). The wheel's
Blackwell-oriented build may carry a cuDNN algorithm enumeration path that is unsafe on Ada
Lovelace (sm_89) hardware.

For the full isolation report including the complete crash sequence, run history across five
wheel versions, minimal reproducer script, and suspect ops, see:
`/home/smola/mxnet/d2l-issues/fine-tuning.md`.

**PyTorch equivalent passes** (loss 0.211, train acc 0.922, test acc 0.949) using
`torchvision.models.resnet18` without `hybridize()` and without cuDNN autotuning.

### Minimal reproducer

```python
"""
Minimal MXNet fine-tuning crash repro.
No d2l, no notebook infrastructure required.
Tests: gluon.model_zoo ResNet-18 pretrained load + hybridize + GPU forward pass.
Expected on broken wheel: process exits with signal 11 or cuDNN autotune segfault.
Expected on fixed wheel: prints output shape and PASS.
"""
import os
os.environ.setdefault("MXNET_CUDNN_AUTOTUNE_DEFAULT", "1")  # keep autotune on

import mxnet as mx
from mxnet import gluon, init, np, npx

npx.set_np()
device = mx.gpu(0)

pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)

finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())

finetune_net.reset_ctx([device])
finetune_net.hybridize()

x = np.random.uniform(size=(4, 3, 224, 224), ctx=device)
y = finetune_net(x)
print("output shape:", y.asnumpy().shape)
print("PASS")
```

Run with:
```bash
CUDA_VISIBLE_DEVICES=0 python repro_fine_tuning.py
# Also try with autotune disabled to distinguish missing kernel vs. autotune bug:
CUDA_VISIBLE_DEVICES=0 MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python repro_fine_tuning.py
```

### Suspect ops / functions

1. **`cudnn_ops.cc:430` — cuDNN autotune dispatch**: crash occurs after two autotune log lines with
   no exception; points to a native abort/segfault during algorithm selection for ResNet-18 conv
   layers on sm_89. The Blackwell-oriented wheel may carry an unsafe algorithm enumeration path for Ada.
2. **`gluon.model_zoo.vision.resnet18_v2(pretrained=True)`**: weights downloaded to CPU then
   transferred to GPU via `reset_ctx`; a broken weight layout or dtype mismatch could trigger a cuDNN
   error on first use.
3. **`net.hybridize()` + first batch dispatch**: deferred compilation means the crash cannot surface
   until `net(X)` is called with a real GPU tensor; the 83 s timing is consistent with model download
   completing before hitting the autotune crash.

---

## Issue 4: `mxnet.__version__` — stale version string (RESOLVED in .20260523.6)

In earlier wheels (`.20260518.1` through `.20260522.2`), `mxnet.__version__` at runtime
reported `'2.0.0+cu13.bw.20260518.1'` regardless of which wheel was actually installed,
because the version string was baked into the build before the wheel-naming step. This
made the in-Python version string useless for telling apart successive builds, and polluted
notebook provenance output.

**As of wheel `.20260523.6`, this is fixed.** `mx.__version__` now correctly reports
`'2.0.0+cu13.bw.20260523.6'` at runtime. The likely fix was regenerating the version-string
header (`include/mxnet/base.h` or equivalent `MXNET_VERSION` source) from the same source as
the wheel-naming script on every build. No action required; documented here as a regression
target should future builds regress.

---

## Issue 5: Noisy "Using Pooled (Naive) StorageManager" log line

On the **first allocation** in each device context, a banner appears on stderr/stdout:

```
[02:51:45] /home/smola/mxnet/src/storage/storage.cc:202: Using Pooled (Naive) StorageManager for GPU
[02:51:45] /home/smola/mxnet/src/storage/storage.cc:202: Using Pooled (Naive) StorageManager for CPU
```

This shows up in every executed d2l-neu notebook's first output cell. It isn't an error, but:

- It pollutes the rendered HTML book and PDFs with internal-source path references.
- It defeats `tools/inject_outputs.py` deduplication when the timestamp is non-deterministic.

Suggested fix: gate the message behind `MXNET_LOG_VERBOSITY` (or an equivalent flag) and leave
it off by default. Print at most once per process, not per device.

Related: `bert-dataset` and `natural-language-inference-and-dataset` also emit
"Storage type fallback detected: operator=stack" warnings (a different but related noise
source from the same storage subsystem).

---

## Issue 6: `chapter_optimization/lr-scheduler` — Gluon Trainer lr-schedule produces ~2× higher loss than PyTorch / JAX

### Description

The `lr-scheduler` notebook trains a LeNet-style CNN on Fashion-MNIST for 30 epochs under
four schedule types (constant, square-root, multi-factor step, and cosine with warmup). The
MXNet implementation uses `mxnet.gluon.Trainer` with its built-in `lr_scheduler` interface
(`lr_scheduler.MultiFactorScheduler`, `lr_scheduler.CosineScheduler`). The final training
losses reported in the cross-framework audit (2026-05-23) are:

| Framework | Final train loss (run 1) | Final train loss (run 2) | Test acc |
|-----------|------------------------:|------------------------:|--------:|
| **MXNet** | **0.353**               | **0.364**               | **0.867–0.870** |
| PyTorch   | 0.171                   | 0.174                   | 0.903–0.904 |
| JAX       | 0.102                   | 0.115                   | 0.908–0.911 |

The MXNet train loss is approximately **2× higher than PyTorch** and **3× higher than JAX**
at the same epoch count with nominally equivalent hyperparameters. Test accuracy is 3–4
percentage points below both baselines — a meaningful convergence regression, not run-to-run
noise.

The root cause is almost certainly a mismatch in how `mxnet.lr_scheduler.MultiFactorScheduler`
and `lr_scheduler.CosineScheduler` interpret their step argument vs. how PyTorch's
`torch.optim.lr_scheduler.MultiStepLR` and the book's custom `CosineScheduler` apply steps.
The key difference: Gluon's built-in schedulers count **update steps** (i.e., minibatch calls
to `trainer.step()`), while the PyTorch and JAX schedulers in the source call `.step()` once
per **epoch**. With 235 minibatches per epoch and 30 epochs, "step 15" in Gluon fires after
~15 minibatches (epoch 0), whereas "milestone 15" in PyTorch fires after 15 full epochs —
roughly a 15× difference in when the first LR drop occurs. The Gluon cosine schedule's
`max_update=20` similarly runs out of decay budget after only ~0.09 epochs, leaving 29.9
epochs at the minimum learning rate.

### Reproducer (MXNet-only, no d2l import)

```python
"""
Repro for lr-scheduler convergence regression in MXNet vs PyTorch.
Trains a small CNN on FashionMNIST with Gluon Trainer + MultiFactorScheduler.
Expected: much higher loss than the equivalent PyTorch MultiStepLR run at epoch 30.
"""
import mxnet as mx
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
from mxnet import lr_scheduler as lrs
import torchvision  # only for dataset; remove if unavailable, use any MNIST loader
import numpy

npx.set_np()
device = mx.gpu(0)

# Build LeNet-style CNN (matching d2l lr-scheduler.md mxnet tab)
net = nn.HybridSequential()
net.add(
    nn.Conv2D(6, kernel_size=5, padding=2, activation='relu'),
    nn.MaxPool2D(2, 2),
    nn.Conv2D(16, kernel_size=5, activation='relu'),
    nn.MaxPool2D(2, 2),
    nn.Dense(120, activation='relu'),
    nn.Dense(84, activation='relu'),
    nn.Dense(10),
)
net.hybridize()
net.initialize(init.Xavier(), ctx=device)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

# ---- CASE A: Gluon built-in MultiFactorScheduler (STEP-based) ----
# "step=[15, 30]" means fire at update #15 and update #30 — 15 minibatches each
scheduler_mx = lrs.MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'lr_scheduler': scheduler_mx})
print("Gluon MultiFactorScheduler: step=[15,30] (step = minibatch update index)")
print("LR at update 0:", scheduler_mx(0))
print("LR at update 14:", scheduler_mx(14))
print("LR at update 15:", scheduler_mx(15))   # drops here — ~epoch 0, batch 15
print("LR at update 30:", scheduler_mx(30))   # drops again — still epoch 0
print()

# ---- CASE B: Manual epoch-level schedule matching PyTorch MultiStepLR ----
# PyTorch milestones=[15, 30] means drop LR at the END of epoch 15 and epoch 30
num_batches = 235  # FashionMNIST train at batch_size=256
print("PyTorch MultiStepLR: milestones=[15,30] (milestone = epoch index)")
print("First LR drop: after epoch 15 =", 15 * num_batches, "update steps")
print("Second LR drop: after epoch 30 =", 30 * num_batches, "update steps")
print()

# ---- CASE C: Gluon CosineScheduler with max_update=20 ----
scheduler_cos = lrs.CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
print("Gluon CosineScheduler: max_update=20 (update steps, not epochs)")
print("  Decay exhausted at update step 20 — which is ~0.09 epochs of FashionMNIST")
print("  Updates per epoch:", num_batches)
print("  LR at update 0:", scheduler_cos(0))
print("  LR at update 10:", scheduler_cos(10))
print("  LR at update 20:", scheduler_cos(20))
print("  LR at update 1000:", scheduler_cos(1000),
      " <- pinned at final_lr for remaining 29+ epochs")
```

### Symptom vs. PyTorch behaviour

PyTorch uses `torch.optim.lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)` and calls `scheduler.step()` once per epoch in the outer training loop. The LR drops at epoch boundaries 15 and 30 as intended. MXNet's `lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5)` counts minibatch update calls — `step=[15, 30]` fires after 15 and 30 *minibatches*, not 15 and 30 epochs. Similarly, `lr_scheduler.CosineScheduler(max_update=20, ...)` exhausts its decay budget after 20 minibatches (~0.09 epochs), pinning at `final_lr=0.01` for the remaining ~29.9 epochs of training.

The net effect is that Gluon's schedule trains at a prematurely small learning rate for nearly all 30 epochs, producing the ~2× higher final loss. PyTorch and JAX control LR at epoch granularity using either a wrapped scheduler or a manual `scheduler(epoch)` call at the end of each epoch loop.

### Suspect ops / functions

- `mxnet.lr_scheduler.MultiFactorScheduler.__call__` — step argument semantics (update steps vs. epochs)
- `mxnet.lr_scheduler.CosineScheduler.__call__` — `max_update` in update steps, not epochs
- `mxnet.gluon.Trainer.__init__` — no conversion or normalization of epoch-level milestones to step-level

---

## Issue 7: `chapter_computer-vision/fcn` — MXNet test accuracy ~15% below PyTorch, loss ~3× higher

### Description

The Fully Convolutional Network (FCN) notebook trains for 5 epochs on the Pascal VOC 2012
semantic segmentation subset. The cross-framework audit (2026-05-23) reports:

| Framework | Train loss | Train acc | Test acc |
|-----------|----------:|----------:|--------:|
| **MXNet** | **1.283**  | **0.721** | **0.723** |
| PyTorch   | 0.417      | 0.870     | 0.854 |
| JAX       | ~1.3–1.8 per epoch (no final acc reported) | — | — |

The MXNet test accuracy gap vs. PyTorch is **(0.854 − 0.723) / 0.854 ≈ 15.3%**. The loss
gap is **3×** (1.283 vs 0.417) with identical hyperparameters (5 epochs, lr=0.001, wd=1e-3,
same SGD optimizer). The notebook completed without errors; this is a training-quality
regression, not a crash.

The most likely cause is a **reduction-scale difference** between
`gluon.loss.SoftmaxCrossEntropyLoss(axis=1)` and PyTorch's `F.cross_entropy`. PyTorch's FCN
source defines the loss as:

```python
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
```

This computes a per-pixel CE loss, then averages over the spatial height dimension, then
averages over width — effectively a mean-over-pixels loss that yields a loss value on the
order of `log(num_classes)` ≈ log(21) ≈ 3.04 at initialization. Gluon's
`SoftmaxCrossEntropyLoss(axis=1)` applies `reduction='mean'` by default over the batch
dimension, but its default behaviour on a 4-D input `(N, C, H, W)` with `axis=1` may
differ in whether and how it normalises over the spatial (H, W) pixels. If Gluon's default
sums over the spatial pixels and divides only by batch size (rather than by N×H×W), the
effective per-example loss is inflated by H×W ≈ 320×480 = 153 600× and the gradient
magnitude is correspondingly larger, potentially causing instability or premature saturation.
Alternatively, if Gluon sums over spatial pixels without normalising, the gradient per step
is proportionally larger, effectively acting as a much higher learning rate for the spatial
loss terms.

### Reproducer (MXNet vs. PyTorch reduction comparison)

```python
"""
Demonstrates the cross-entropy reduction difference between
gluon.loss.SoftmaxCrossEntropyLoss(axis=1) and torch F.cross_entropy
on a tiny semantic-segmentation-shaped tensor.
No dataset required.
"""
import numpy as _np

# --- MXNet ---
import mxnet as mx
from mxnet import gluon, np as mnp, npx
npx.set_np()

N, C, H, W = 2, 21, 8, 8  # tiny FCN batch: 2 images, 21 classes, 8×8 spatial
rng = _np.random.default_rng(0)

logits_np = rng.standard_normal((N, C, H, W)).astype('float32')
labels_np  = rng.integers(0, C, size=(N, H, W)).astype('int32')

logits_mx = mnp.array(logits_np, ctx=mx.cpu())
labels_mx = mnp.array(labels_np, ctx=mx.cpu())

loss_gluon = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
mx_loss = loss_gluon(logits_mx, labels_mx)
print("Gluon SoftmaxCrossEntropyLoss(axis=1):")
print("  output shape:", mx_loss.shape)
print("  mean value:", float(mx_loss.mean().asnumpy()))
print("  note: if shape is (N,) the value is summed over H×W pixels per sample")

# --- PyTorch ---
import torch
import torch.nn.functional as F

logits_pt = torch.tensor(logits_np)
labels_pt = torch.tensor(labels_np.astype('int64'))

# PyTorch FCN source: reduction='none', then .mean(1).mean(1) → mean over H, then W
pt_loss_spatial = F.cross_entropy(logits_pt, labels_pt, reduction='none').mean(1).mean(1)
print("\nPyTorch F.cross_entropy(reduction='none').mean(1).mean(1):")
print("  output shape:", pt_loss_spatial.shape)
print("  mean value:", float(pt_loss_spatial.mean()))

# Also show PyTorch with reduction='mean' over all dims for comparison
pt_loss_mean = F.cross_entropy(logits_pt, labels_pt, reduction='mean')
print("\nPyTorch F.cross_entropy(reduction='mean'):")
print("  scalar value:", float(pt_loss_mean))
print("\nRatio (Gluon mean / PT mean-over-all):", float(mx_loss.mean().asnumpy()) / float(pt_loss_mean))
```

### Symptom vs. PyTorch behaviour

At random initialisation the expected per-pixel cross-entropy is `log(21) ≈ 3.04`. PyTorch's
FCN loss averages over H×W pixels and reports a value near 3.04 per image, which is correct.
If Gluon's loss instead sums over H×W pixels (for an 8×8 spatial grid: 64 pixels), the
reported loss is ~64× larger at the same convergence point, and the gradient passed back to
the network is proportionally inflated. This would appear as a falsely high loss throughout
training even when pixel-wise accuracy is reasonable, matching the observed 1.283 vs. 0.417
discrepancy.

### Suspect ops / functions

- `gluon.loss.SoftmaxCrossEntropyLoss.__init__` — default `reduction` and `weight` behaviour on 4-D inputs with `axis=1`
- The internal reduction over the spatial axes (H, W) after the per-pixel softmax: whether it is `sum` or `mean`
- `gluon.loss.SoftmaxCrossEntropyLoss` source in `src/operator/numpy/np_loss_op.cc` or the Gluon Python wrapper

---

## Cross-framework observations (not bugs)

**Recommender-systems chapter (mxnet-only):** All notebooks in
`chapter_recommender-systems/` (autorec, ctr-prediction, deepfm, fm, movielens, neumf, seernet)
are MXNet-exclusive across the entire book — no PyTorch or JAX siblings exist. Differences
from a non-existent baseline cannot be flagged as regressions. These notebooks all pass on the
current wheel.

**`chapter_recurrent-modern/deep-rnn` — trainer.fit() workaround removed (RESOLVED):**
An earlier revision of the source commented out `trainer.fit()` for the MXNet tab with a note
that the notebook took >1 hour to execute and was excluded from CI. This workaround has been
removed from the current source. On wheel .20260523.6 the notebook runs successfully in **415 s**
(~7 minutes) using GPU acceleration. Execution stamp:
`_notebooks/mxnet/chapter_recurrent-modern/deep-rnn.executed` (2026-05-23 16:31).

**Storage-type fallback warnings in NLP pretraining:** `chapter_natural-language-processing-pretraining/bert-dataset` and `chapter_natural-language-inference-and-dataset` emit
"Storage type fallback detected: operator=stack" warnings. These are non-fatal and related to
Issue 5 (the storage-manager noise). The `stack` operator lacks a sparse-storage implementation
and falls back to dense; the warning fires once per affected operator per process.

**Spurious "GPU context requested, but no GPUs found" stderr in CPU-only cells:** Several
notebooks (mlp, numerical-stability-and-init, conv-layer, autograd, calculus, multivariable-calculus,
maximum-likelihood) emit this message on cells that perform only CPU computation. MXNet probes for
available GPUs even when `d2l.try_gpu()` or `npx.gpu(0)` is not called. This adds noise to notebook
output in CPU-fallback environments. Gating the probe behind an explicit device request would
eliminate it.

---

## Things that USED TO fail and now pass

Documented here so the mxnet team has regression-test targets:

| Notebook | Symptom under earlier wheel | Last broken wheel |
|---|---|---|
| `chapter_computer-vision/ssd.ipynb` | IndexError via argmax size-1 GPU bug (Issue 1) | `.20260522.x` |
| `chapter_natural-language-processing-applications/natural-language-inference-bert.ipynb` | DeadKernelError at ~1095 s | `.20260522.x` |
| `chapter_convolutional-neural-networks/channels.ipynb` | AssertionError on CPU | `.20260518.2` |
| ~120 other GPU notebooks | `cudaErrorNoKernelImageForDevice` (error 209) | `.20260518.2` |

The progression from `error 209` → USE_OPENCV error → partial sm_89 coverage in earlier
wheels is documented in `docs/mxnet-runtime-diagnostics.md` in the d2l-neu repo.
