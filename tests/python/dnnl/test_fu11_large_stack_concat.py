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

"""FU-11 regression: oneDNN concat/stack with > 512 sources falls back cleanly.

Before this fix, `np.stack` of >512 1-D arrays (the path used by
`gluon.data.DataLoader`'s `Stack` batchify whenever `batch_size > 512`)
dispatched to oneDNN's concat primitive, which then failed in
`cvt_primitive_args` with "bad number of inputs (expected N+1 got N)".

The fix gates `SupportDNNLStack` and `SupportDNNLConcat` on input count,
falling back to the generic CPU implementation for wide inputs.
"""
import os
import pytest
import numpy as _np
from mxnet import np, npx
import mxnet as mx

npx.set_np()


# Boundary sizes around the known oneDNN failure (513 inputs).
# Test 256/512 to confirm DNNL path still works; 513/1024 to confirm
# fallback works.
SIZES_DNNL = [16, 64, 256]            # values <= kDNNLStackMaxInputs (DNNL path)
SIZES_FALLBACK = [257, 513, 1024]     # values > kDNNLStackMaxInputs (fallback path)
ALL_SIZES = SIZES_DNNL + SIZES_FALLBACK


@pytest.mark.parametrize("n", ALL_SIZES)
def test_fu11_stack_1d_int32_slices(n):
    """Stack n 1-D int32 row-slices from a 2-D array.

    Mirrors what `gluon.data.DataLoader` batchify does for an
    `ArrayDataset(X, Y)` with `batch_size=n`.
    """
    rows, cols = max(n + 100, 2048), 32
    big = np.array(_np.arange(rows * cols, dtype=_np.int32).reshape(rows, cols))
    big.wait_to_read()
    arrs = [big[i] for i in range(n)]
    out = np.stack(arrs)
    out.wait_to_read()
    assert out.shape == (n, cols)
    # Spot-check correctness: row i of out should equal big[i]
    out_np = out.asnumpy()
    big_np = big.asnumpy()
    _np.testing.assert_array_equal(out_np[0], big_np[0])
    _np.testing.assert_array_equal(out_np[-1], big_np[n - 1])


@pytest.mark.parametrize("n", ALL_SIZES)
def test_fu11_stack_1d_fp32(n):
    """Stack n 1-D float32 arrays (no slicing — fresh allocations)."""
    arrs = [np.array(_np.arange(32, dtype=_np.float32) + i * 0.1)
            for i in range(n)]
    out = np.stack(arrs)
    out.wait_to_read()
    assert out.shape == (n, 32)
    out_np = out.asnumpy()
    _np.testing.assert_allclose(out_np[0], _np.arange(32, dtype=_np.float32), rtol=1e-6)
    _np.testing.assert_allclose(
        out_np[n - 1], _np.arange(32, dtype=_np.float32) + (n - 1) * 0.1, rtol=1e-6)


@pytest.mark.parametrize("n", ALL_SIZES)
def test_fu11_concat_axis0(n):
    """Concatenate n (1, 32) arrays along axis 0 — same code path as stack."""
    arrs = [np.array(_np.arange(32, dtype=_np.float32).reshape(1, 32) + i * 0.1)
            for i in range(n)]
    out = np.concatenate(arrs, axis=0)
    out.wait_to_read()
    assert out.shape == (n, 32)


def test_fu11_dataloader_batch_1024():
    """End-to-end: `gluon.data.DataLoader` with batch_size=1024 and an
    `ArrayDataset`-style backing.  This exact pattern (large batch + 1-D
    int32 sequences) is what fails in d2l's TimeMachine + RNNLMScratch
    notebook on wheels prior to this fix.
    """
    from mxnet.gluon.data import DataLoader, ArrayDataset
    rows, num_steps = 4096, 32
    X = np.array(_np.random.randint(0, 28, size=(rows, num_steps), dtype=_np.int32))
    Y = np.array(_np.random.randint(0, 28, size=(rows, num_steps), dtype=_np.int32))
    X.wait_to_read(); Y.wait_to_read()
    dataset = ArrayDataset(X, Y)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    seen = 0
    for batch in loader:
        for b in batch:
            b.wait_to_read()
        assert batch[0].shape[1] == num_steps
        seen += batch[0].shape[0]
    assert seen == rows
