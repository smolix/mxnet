# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Regression tests for FU-4: fork-safe oneDNN / DNNL engine.

Before the fix, DataLoader with num_workers > 0 would fork child processes that
inherited a live oneDNN engine state.  The first oneDNN primitive execution in
those children raised 'MXNetError: could not execute a primitive'.

The fix adds an atfork_child() handler that:
  - re-placement-new's the CpuEngine,
  - drops the primitive cache,
  - invalidates NDArray::dnnl_mem_ handles, and
  - flips g_dnnl_forked_child so DNNLEnvSet() returns false in workers.

This file verifies that forked DataLoader workers can complete iteration without
raising any exception.
"""

import pytest
import numpy as np
import mxnet as mx
from mxnet import np as mnp, npx, gluon
from mxnet.gluon.data import DataLoader, ArrayDataset

pytestmark = pytest.mark.usefixtures("set_np_semantics")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _warm_up_dnnl_concat():
    """Trigger oneDNN initialisation via a concat op so the engine is live
    before any fork occurs."""
    a = mnp.zeros((4, 8), dtype='float32', ctx=mx.cpu())
    b = mnp.ones((4, 8), dtype='float32', ctx=mx.cpu())
    out = mnp.concatenate([a, b], axis=0)
    out.wait_to_read()


def _warm_up_dnnl_conv():
    """Heavier warmup: run a small conv2d through the DNNL path."""
    layer = gluon.nn.Conv2D(channels=8, kernel_size=3, padding=1, in_channels=4)
    layer.initialize(ctx=mx.cpu())
    x = mnp.random.uniform(size=(2, 4, 16, 16), ctx=mx.cpu()).astype('float32')
    for _ in range(3):
        y = layer(x)
        y.wait_to_read()


def _make_loader(num_workers, batch_size=8, n_samples=64):
    data = np.random.uniform(size=(n_samples, 8)).astype('float32')
    labels = np.arange(n_samples, dtype='float32')
    ds = ArrayDataset(data, labels)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, try_nopython=False)


# ---------------------------------------------------------------------------
# 1. Basic fork-safety: 4 workers, minimal oneDNN warmup
# ---------------------------------------------------------------------------

def test_dataloader_num_workers_4_no_primitive_failure():
    """Repro for FU-4: DataLoader fork with 4 workers must not raise
    'could not execute a primitive' (the symptom on master pre-fix).

    We pre-touch oneDNN in the parent so the engine is initialised before the
    fork, which is the exact scenario that triggered the original bug.
    """
    _warm_up_dnnl_concat()

    loader = _make_loader(num_workers=4)
    iters = 0
    # Iterate — if any worker raises, the DataLoader propagates the exception here.
    for x, y in loader:
        x.wait_to_read()
        y.wait_to_read()
        iters += 1

    assert iters == 8, f"Expected 8 batches, got {iters}"


# ---------------------------------------------------------------------------
# 2. Heavier oneDNN warmup (conv2d) before fork, 2 workers
# ---------------------------------------------------------------------------

def test_dataloader_num_workers_2_with_dnnl_op_in_parent():
    """Heavier oneDNN warmup (conv2d primitive) before fork.

    Exercises a richer engine state: weight reorder primitive, output buffer
    allocation, and the DNNL primitive cache — all inherited by forked workers.
    """
    _warm_up_dnnl_conv()

    loader = _make_loader(num_workers=2)
    iters = 0
    for x, y in loader:
        x.wait_to_read()
        y.wait_to_read()
        iters += 1

    assert iters == 8, f"Expected 8 batches, got {iters}"


# ---------------------------------------------------------------------------
# 3. Exhaust the loader twice (catches one-shot atfork bugs)
# ---------------------------------------------------------------------------

def test_dataloader_num_workers_4_repeated_iter():
    """Iterate through the DataLoader twice.

    Some one-shot atfork implementations only protect the first pass; a second
    iteration re-enters the worker pool and can expose lingering state bugs.
    """
    _warm_up_dnnl_concat()

    loader = _make_loader(num_workers=4, n_samples=32, batch_size=4)

    for pass_idx in range(2):
        iters = 0
        for x, y in loader:
            x.wait_to_read()
            y.wait_to_read()
            iters += 1
        assert iters == 8, (
            f"Pass {pass_idx + 1}: expected 8 batches, got {iters}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
