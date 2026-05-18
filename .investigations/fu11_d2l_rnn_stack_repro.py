"""
FU-11 minimal reproducer: d2l RNN/embedding notebooks fail with
`MXNetError: could not execute a primitive` at the DataLoader's
`Stack` batchify, AFTER constructing a `d2l.Trainer` and calling
`trainer.fit(model, data)`.

Key finding (2026-05-18 22:18): the failure is NOT a fork issue
(`d2l.DataModule.get_tensorloader` returns a DataLoader with
`num_workers=0`).  Iterating the same DataLoader directly — without
constructing a Trainer or model — PASSES.  The Trainer setup
mutates some CPU oneDNN process-global state that subsequently
breaks `np.stack`.

To reproduce the failure path (wheel `2.0.0+cu13.bw.20260518.2`):

    cd /workspace/d2l-neu
    CUDA_VISIBLE_DEVICES=0 MXNET_BACKTRACE=1 MXNET_ENGINE_TYPE=NaiveEngine \\
        .venv-mxnet/bin/python -m jupyter nbconvert \\
        --to notebook --execute --inplace \\
        --ExecutePreprocessor.kernel_name=python3 \\
        --ExecutePreprocessor.timeout=60 \\
        --output /tmp/gru_repro.ipynb \\
        _notebooks/mxnet/chapter_recurrent-modern/gru.ipynb

To confirm the workaround:

    MXNET_ONEDNN_ENABLED=0 ...same command... → trains successfully.

To localise the failure to `_api_internal.stack`:
look at the NaiveEngine traceback — it points at:
    mxnet/gluon/data/batchify.py:95 in Stack.__call__
    → mxnet.numpy.stack(arrays, axis, out)
    → _api_internal.stack(*arrays, axis, out)  ← raises

Direct-iteration control (should PASS, confirming the DataLoader
itself is fine when used in isolation):
"""
import os
os.environ.setdefault("MXNET_ENGINE_TYPE", "NaiveEngine")
os.environ.setdefault("MXNET_BACKTRACE", "1")

from d2l import mxnet as d2l
from mxnet import npx
npx.set_np()

data = d2l.TimeMachine(batch_size=32, num_steps=32)
print(f"data built, X.shape={data.X.shape}, Y.shape={data.Y.shape}")

loader = data.train_dataloader()
print(f"DataLoader.num_workers = {loader._num_workers}")
assert loader._num_workers == 0, "Test invariant: NOT a fork issue"

# This part PASSES — DataLoader iteration without Trainer is fine.
for i, batch in enumerate(loader):
    print(f"batch {i}: shapes {[b.shape for b in batch]}")
    if i >= 2:
        break
print("Direct DataLoader iteration: PASS (no Trainer involved)")
print()
print("Failure path requires the full nbconvert run above.")
