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

"""Regression test for the 0-dim NDArray oneDNN crash that bit ~24 d2l
notebooks against the 2026-05-22 cleanup wheel.

The crash was in `NDArray::Chunk::SetDNNLMem` (src/ndarray/ndarray.cc:624):
when an NDArray reached the oneDNN binding path with `shape.ndim() == 0`
(scalar — common after `mx.np.<arr>.sum()`, `mean()`, `.item()`, etc.)
the binding would `LOG(FATAL)` with "oneDNN doesn't support 0 dimensions".

The fix maps 0-dim NDArrays to a 1-D length-1 oneDNN memory descriptor
(same byte layout) so transfers go through cleanly; MXNet's own
shape_ field is the source of truth for caller-visible shape.

This test guards both the imperative-NumPy and classic-NDArray paths.
"""

import numpy as np
import pytest

import mxnet as mx


def test_np_sum_item_does_not_crash():
    # The canonical d2l failure: a full-reduction sum returns a 0-dim
    # NDArray and calling .item() routes through asnumpy / oneDNN
    # binding.  Must not LOG(FATAL).
    x = mx.np.array([1.0, 2.0, 3.0, 4.0])
    s = x.sum()
    assert s.shape == ()
    assert s.ndim == 0
    assert s.item() == pytest.approx(10.0)


def test_np_sum_asnumpy_does_not_crash():
    x = mx.np.array([1.5, 2.5, 3.5])
    s = x.sum()
    arr = s.asnumpy()
    assert arr.shape == ()
    assert float(arr) == pytest.approx(7.5)


def test_np_zero_dim_arithmetic_round_trip():
    # Chain a few ops on 0-dim NDArrays, exercising both ways through
    # asnumpy / item().
    a = mx.np.array([1.0, 2.0, 3.0])
    b = mx.np.array([4.0, 5.0, 6.0])
    scalar = (a * b).sum() / a.sum()
    # 0-dim scalar after the division.
    assert scalar.shape == ()
    expected = float(np.array([1.0, 2.0, 3.0]).dot([4.0, 5.0, 6.0])) / 6.0
    assert float(scalar) == pytest.approx(expected)


def test_np_mean_dot_squared_norm():
    # mxnet.gluon.utils.clip_global_norm and similar utilities use
    # `_mx_np.square(x).sum().item()` in a loop.  Make sure the same
    # idiom does not crash.
    rng = np.random.RandomState(0)
    for shape in [(4,), (3, 4), (2, 3, 4)]:
        np_x = rng.randn(*shape).astype('float32')
        x = mx.np.array(np_x)
        sq_sum = mx.np.square(x).sum().item()
        np_sq_sum = float((np_x.astype(np.float64) ** 2).sum())
        assert sq_sum == pytest.approx(np_sq_sum, rel=1e-5)


def test_nd_sum_keeps_one_dim_shape():
    # The classic NDArray API has always returned shape (1,) instead of
    # a 0-dim scalar for full-reductions; preserve that contract.
    x = mx.nd.array([1.0, 2.0, 3.0, 4.0])
    s = x.sum()
    assert s.shape == (1,)
    assert s.asscalar() == pytest.approx(10.0)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
