# Threading & OOM bug log

Captures everything that's been narrowed down on the two open
d2l-issue-3-class bugs so a fresh session can resume without
re-bisecting.

## Bug 1 — ThreadedEnginePerDevice CPU→GPU NDArray handle race

### What

Multi-GPU d2l training notebooks that use any CPU `kRandom`-using
image-augmentation op (RandomResizedCrop, RandomFlipLeftRight,
RandomBrightness, etc.) in a `gluon.data.DataLoader` pipeline
SEGV after the first training stage's eval pass or after a
`finetune→scratch` train_fine_tuning transition. The crashing thread
is the main Python thread inside `MXNDArraySyncCopyToCPU` (i.e.,
`asnumpy()`).

### Affected notebooks (observed across multiple `~/d2l-neu/logs/`)

- `chapter_computer-vision/fine-tuning.ipynb`
- `chapter_convolutional-modern/alexnet.ipynb`
- `chapter_convolutional-modern/cnn-design.ipynb`
- `chapter_convolutional-modern/densenet.ipynb`
- `chapter_computer-vision/ssd.ipynb`

### Bisection ladder (all PASS unless noted)

Repros live under `~/mxnet/d2l-issues/repro_*.py`. Build:
`mxnet-2.0.0+cu13.bw.20260523.8` (post `74084a529`, contains
defense-in-depth Random<cpu>::mutex).

| Configuration | Engine | Result |
|---|---|---|
| Random GPU tensors, no DataLoader, 2 sequential trainings | ThreadedEnginePerDevice | PASS |
| + CPU NDArrays + `split_and_load` | ThreadedEnginePerDevice | PASS |
| + 5-epoch + `evaluate_accuracy_gpus` between epochs | ThreadedEnginePerDevice | PASS |
| + Gluon `DataLoader` over a synthetic Dataset (no OpenCV) | ThreadedEnginePerDevice | PASS |
| + `ImageFolderDataset` + identity transform | ThreadedEnginePerDevice | PASS |
| + `ImageFolderDataset` + ANY `npx.image.random_*` op (STAGE 1+) | ThreadedEnginePerDevice | **CRASH** |
| Same trigger | NaiveEngine | PASS |
| Same trigger | ThreadedEngine (non-PerDevice via `MXNET_ENGINE_TYPE=ThreadedEngine`) | PASS |

### Fault signature (clean trace under `OMP_NUM_THREADS=1`, `OPENCV_NUM_THREADS=1`)

```
Fatal Python error: Segmentation fault

Thread 0x...... (most recent call first):
  File "mxnet/ndarray/ndarray.py", line 2655 in asnumpy
  File "run_aug_chain.py", line 120 in train
```

`ndarray.py:2655` is:
```python
check_call(_LIB.MXNDArraySyncCopyToCPU(self.handle, ...))
```

Single thread, no native stack frames after the C boundary. The
NDArray's storage handle has been freed by the time `asnumpy` runs.

### Where the bug lives

`src/engine/threaded_engine_perdevice.cc`. The CPU-worker thread that
produced the NDArray completes, the engine moves on to the GPU
consumer (a `split_and_load` copy from CPU host memory to GPU), and
somewhere in that hand-off the storage Chunk's reference count
reaches 0 before the GPU consumer finishes. ThreadedEngine
(non-PerDevice) does not exhibit this — its dependency tracking
keeps the producer's outputs alive.

### What's already been tried (didn't fix)

1. **`MXNET_CUDNN_AUTOTUNE_DEFAULT=0`** — does not change outcome.
2. **`MXNET_CUDNN_AUTOTUNE_SERIALIZE=1`** (in `.7`/`.8`) — does not
   change outcome.
3. **`Random<cpu>::mutex()`** lock added in commit `74084a529`
   covering 10 CPU image-aug FCompute sites (defense-in-depth) — does
   not change outcome.

### Currently-working workaround (shipped in `.8` only as documentation)

```bash
MXNET_ENGINE_TYPE=ThreadedEngine
```

Set in the user's environment or in the d2l notebook driver. Verified
clean on:

- `d2l-issues/repro_aug_chain.py` `AUG_STAGE=4` (10/10 epochs)
- `_notebooks/mxnet/chapter_computer-vision/fine-tuning.ipynb`
  (real notebook, multi-GPU 4× RTX 4090)

### Next investigation steps

1. **Look at `src/engine/threaded_engine_perdevice.cc`** —
   specifically how a completed op's `var` is dropped from the
   dependency graph and how the consumer chain on a different device
   keeps producer-side storage alive.
2. **Compare with `ThreadedEngine::ExecuteOprFn`** (the non-PerDevice
   path that works) — look for missing `shared_ptr` increment in the
   PerDevice variant when an op result is consumed by a different
   device's worker.
3. **Sanitize under TSAN** — `cmake -DUSE_TSAN=ON` followed by the
   repro_aug_chain.py STAGE=1 script. Should pinpoint the unprotected
   shared write.

### Files of interest

- `src/engine/threaded_engine.h` — base class, op scheduling
- `src/engine/threaded_engine.cc` — base impl
- `src/engine/threaded_engine_perdevice.cc` — the variant that
  exhibits the bug
- `src/engine/threaded_engine_pooled.cc` — for reference, what is
  NOT broken
- `include/mxnet/ndarray.h` — NDArray's `Chunk` shared_ptr lifecycle
- `src/ndarray/ndarray.cc:SyncCopyToCPU` — the crash site

---

## Bug 2 — sentiment-analysis-rnn cross-process OOM at 24.7 GB pool

### What

`chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb`
OOMs at ~82s into training when a co-tenant `bert-pretraining` notebook
is running on the same physical GPU (the d2l scheduler runs
`GPU_SLOTS=8` on 4× 24 GB GPUs → two notebooks per GPU). The
exact error:

```
MXNetError: Memory allocation failed out of memory
(requested 14680064 bytes, pool used 24728231936, device free 1028915 ...)
```

`pool used = 24.7 GB` on a 24.5 GB GPU — MXNet's allocator has cached
basically all of VRAM in its pool. The 14 MB request can't be served
even after the retry-with-backoff path.

### Affected scenarios

Only reproducible under cross-process contention (two MXNet processes
on the same GPU). Solo runs of `sentiment-analysis-rnn` pass cleanly.

### What's already in `.6`/`.7`/`.8`

`src/storage/pooled_storage_manager.h::Alloc` has a retry-with-backoff
loop for `cudaErrorMemoryAllocation`:

- Default: 4 retries × 50ms / 100ms / 200ms / 400ms (≤750 ms wall
  before FATAL)
- Each retry calls `cudaDeviceSynchronize()` + `ReleaseAllNoLock(false)`
- Knobs: `MXNET_GPU_MEM_POOL_OOM_RETRIES`,
  `MXNET_GPU_MEM_POOL_OOM_BACKOFF_MS`
- Diagnostic FATAL message includes requested bytes / pool used /
  device free/total / retries

### Why the retry didn't catch this case

`pool used = 24.7 GB / device free 1 MB` means **this process's own
pool** already holds basically all the memory. There's no room for
the co-tenant BERT-pretraining process to give up — even if it freed
its entire arena, this process can't see the freed VRAM until OS
re-issues it.

The retry path calls `ReleaseAllNoLock(false)` (the pool's own
release-on-idle) but that only frees blocks not currently in use by
in-flight ops. The pool's high-water mark from peak training
activity persists.

### Hypotheses

1. **Per-arena fragmentation growth.** The textCNN-then-RNN model's
   training pattern (variable-length IMDB sequences → variable-sized
   activations) inflates the pool's high-water mark step by step.
   Free blocks at the wrong size remain unused; the pool grows.
   Test: run the notebook solo, watch `nvidia-smi` for the pool's
   peak. If it approaches 24 GB even solo, this is the cause.
2. **`MXNET_GPU_MEM_POOL_TYPE` defaults.** Default is `Naive`. The
   `Round` strategy buckets allocations to power-of-2 sizes and may
   fragment less. Quick env test:
   ```bash
   MXNET_GPU_MEM_POOL_TYPE=Round
   MXNET_GPU_MEM_POOL_PAGE_SIZE=2097152
   ```
3. **Sliding-window LSTM hidden state retention.** Sequence RNN
   workloads sometimes retain hidden states across batches; under
   `bidirectional=True` + 2 layers, the cached states can balloon.
   Compare `mx.gluon.rnn` cache behavior `.5 → .8` for ABI changes.

### Currently-working workarounds (suggestions, untested in this session)

- Bump retries:
  ```bash
  MXNET_GPU_MEM_POOL_OOM_RETRIES=10
  MXNET_GPU_MEM_POOL_OOM_BACKOFF_MS=200
  ```
  Total budget then is 10 × (200ms, 400ms, …) ≈ 5–10 s. Should be
  enough to outlast BERT-pretraining's transient peaks.
- Switch pool strategy:
  ```bash
  MXNET_GPU_MEM_POOL_TYPE=Round
  MXNET_GPU_MEM_POOL_PAGE_SIZE=2097152
  ```

### Next investigation steps

1. **Single-process pool growth profile.** Instrument `Alloc`/`Free`
   in `pooled_storage_manager.h` to log running totals, then run
   sentiment-analysis-rnn solo. Plot pool size vs iteration. If
   monotonic-growing past ~10 GB, the pool is the cause.
2. **Compare `.6` vs `.8` retention.** The d2l agent reports `.6`
   was green on this notebook; `.8` (and `.7`) OOM. Find the lib diff
   between them — likely a defaulted env var or pool-strategy
   behavior change.
3. **Cross-process check.** Capture `nvidia-smi --query-compute-apps`
   snapshots while running both notebooks; quantify how much each
   process holds when one OOMs.

### Files of interest

- `src/storage/pooled_storage_manager.h::Alloc` — current retry loop
- `src/storage/storage.cc` — strategy selection
- `python/mxnet/gluon/rnn/_layers.py` — RNN layer state cache
- `~/d2l-neu/_notebooks/mxnet/chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb` — repro

---

## Status snapshot

| Bug | Reproduces under | Workaround in `.8` | Real fix |
|---|---|---|---|
| 1 (engine race) | default ThreadedEnginePerDevice | `MXNET_ENGINE_TYPE=ThreadedEngine` | needs engine-layer patch in `threaded_engine_perdevice.cc` |
| 2 (cross-process OOM) | sentiment-RNN + BERT-pretrain co-tenant | retry env knob bumps (untested) | needs pool-growth profile |

Both bugs need a fresh session to actually patch. This log is the hand-off.
