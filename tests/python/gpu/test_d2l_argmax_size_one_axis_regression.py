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

"""Regression test for `mxnet.numpy.argmax` on a size-1 reduction axis (GPU).

Triggered the d2l SSD notebook failure: the banana detection dataset has
exactly one ground-truth box per image, so `assign_anchor_to_bbox` computes
`np.argmax(jaccard, axis=1)` with `jaccard.shape == (num_anchors, 1)`.  The
broken GPU `_npi_argmax` returned `[0, 1, 2, ..., num_anchors - 1]` instead
of all zeros, which made `multibox_target` index `label[5362, ...]` on a
`label` of shape `(1, 5)` and raise `IndexError`.

Root cause: `reduce_kernel_M1` (the M==1 fast path in
`src/operator/tensor/reduce_rtc.cc`) used the kernel's outer flat-output
index as `index` in `FUNC = AType(OP(...), index)`, but FUNC expects the
position along the reduction axis (which for M==1 is always 0).  Fix
shadows `index` with a local 0 around the REDUCER.Reduce call.
"""

import numpy as np
import pytest

import mxnet as mx
from mxnet import np as mxnp


def _cuda_available():
    return mx.runtime.Features().is_enabled("CUDA") and mx.context.num_gpus() > 0


pytestmark = pytest.mark.skipif(
    not _cuda_available(), reason="GPU required for this regression")


def _argmax_matches_numpy(np_arr, axis, device):
    arr = mxnp.array(np_arr, ctx=device)
    got = mxnp.argmax(arr, axis=axis).asnumpy()
    expected = np_arr.argmax(axis=axis)
    return got, expected


@pytest.mark.parametrize("axis_size", [1, 2, 3, 5])
def test_np_argmax_axis_1_matches_numpy(axis_size):
    rng = np.random.RandomState(0)
    data_np = rng.rand(8, axis_size).astype('float32')
    got, expected = _argmax_matches_numpy(data_np, axis=1, device=mx.gpu(0))
    np.testing.assert_array_equal(got, expected,
        err_msg=f"argmax(axis=1) mismatch at axis_size={axis_size}")


@pytest.mark.parametrize("axis_size", [1, 2, 3, 5])
def test_np_argmax_axis_0_matches_numpy(axis_size):
    rng = np.random.RandomState(1)
    data_np = rng.rand(axis_size, 8).astype('float32')
    got, expected = _argmax_matches_numpy(data_np, axis=0, device=mx.gpu(0))
    np.testing.assert_array_equal(got, expected,
        err_msg=f"argmax(axis=0) mismatch at axis_size={axis_size}")


def test_np_argmax_size_one_axis_returns_zeros():
    # The exact d2l SSD shape: many anchors, one gt-box.
    data_np = np.random.RandomState(2).rand(5444, 1).astype('float32')
    arr = mxnp.array(data_np, ctx=mx.gpu(0))
    got = mxnp.argmax(arr, axis=1).asnumpy()
    np.testing.assert_array_equal(got, np.zeros(5444, dtype=got.dtype))


def test_np_argmax_keepdims_size_one_axis():
    data_np = np.random.RandomState(3).rand(8, 1).astype('float32')
    arr = mxnp.array(data_np, ctx=mx.gpu(0))
    got = mxnp.argmax(arr, axis=1, keepdims=True).asnumpy()
    assert got.shape == (8, 1)
    np.testing.assert_array_equal(got.ravel(), np.zeros(8, dtype=got.dtype))


def test_np_argmin_size_one_axis_returns_zeros():
    # argmin shares the same kernel infrastructure; cover it explicitly so
    # an asymmetric fix doesn't leave argmin broken.
    data_np = np.random.RandomState(4).rand(16, 1).astype('float32')
    arr = mxnp.array(data_np, ctx=mx.gpu(0))
    got = mxnp.argmin(arr, axis=1).asnumpy()
    np.testing.assert_array_equal(got, np.zeros(16, dtype=got.dtype))


def test_np_argmax_size_one_first_axis():
    # Symmetric coverage: reduce the leading axis when it has size 1.
    data_np = np.random.RandomState(5).rand(1, 12).astype('float32')
    arr = mxnp.array(data_np, ctx=mx.gpu(0))
    got = mxnp.argmax(arr, axis=0).asnumpy()
    np.testing.assert_array_equal(got, np.zeros(12, dtype=got.dtype))


def test_np_argmax_3d_size_one_middle_axis():
    # 3-D shape with the middle axis being the size-1 reduction.
    data_np = np.random.RandomState(6).rand(4, 1, 7).astype('float32')
    arr = mxnp.array(data_np, ctx=mx.gpu(0))
    got = mxnp.argmax(arr, axis=1).asnumpy()
    assert got.shape == (4, 7)
    np.testing.assert_array_equal(got, np.zeros((4, 7), dtype=got.dtype))


# ---- Legacy mx.nd.argmax coverage ----
# The d2l-mxnet-issues.md report flagged that the legacy `mx.nd.argmax` may
# share the broken kernel.  After the reduce_kernel_M1 fix in
# src/operator/tensor/reduce_rtc.cc both API surfaces are covered by the same
# kernel — so we pin the legacy surface too to catch a future split.

@pytest.mark.parametrize("axis_size", [1, 2, 3, 5])
def test_nd_argmax_axis_1_matches_numpy(axis_size):
    rng = np.random.RandomState(7)
    data_np = rng.rand(8, axis_size).astype('float32')
    arr = mx.nd.array(data_np, ctx=mx.gpu(0))
    got = mx.nd.argmax(arr, axis=1).asnumpy().astype(np.int64)
    expected = data_np.argmax(axis=1)
    np.testing.assert_array_equal(got, expected,
        err_msg=f"nd.argmax(axis=1) mismatch at axis_size={axis_size}")


def test_nd_argmax_size_one_axis_returns_zeros():
    data_np = np.random.RandomState(8).rand(5444, 1).astype('float32')
    arr = mx.nd.array(data_np, ctx=mx.gpu(0))
    got = mx.nd.argmax(arr, axis=1).asnumpy().astype(np.int64)
    np.testing.assert_array_equal(got, np.zeros(5444, dtype=got.dtype))


def test_nd_argmin_size_one_axis_returns_zeros():
    data_np = np.random.RandomState(9).rand(16, 1).astype('float32')
    arr = mx.nd.array(data_np, ctx=mx.gpu(0))
    got = mx.nd.argmin(arr, axis=1).asnumpy().astype(np.int64)
    np.testing.assert_array_equal(got, np.zeros(16, dtype=got.dtype))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
