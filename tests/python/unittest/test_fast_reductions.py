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

"""
Correctness tests for the fast-path sum/mean reductions.

These exercise the cases that now have dedicated fast paths -- global
(scalar-output) reductions and outer/leading-axis reductions -- as well as
the generic mixed-axis paths and the kAddTo accumulation path through
autograd. Everything here runs on CPU by default and is checked against
NumPy (reference computed in float64).
"""

import numpy as onp
import pytest
import mxnet as mx
from mxnet import np, npx

npx.set_np()


def _tol(dtype, fast=True):
    # The global and leading-axis fast paths accumulate in double, so f32 is
    # tight there. The generic mixed-axis path accumulates in f32, so it needs a
    # looser bound that reflects single-precision round-off over the reduced run.
    if dtype == onp.float64:
        return 1e-12
    if dtype == onp.float16:
        # Output is stored in float16; round-off is dominated by the cast back.
        return 1e-2
    return 1e-5 if fast else 2e-4


def _atol(dtype, fast=True):
    # Absolute floor so a sum that lands near zero (cancellation) doesn't blow up
    # the *relative* error: a tiny absolute round-off then reads as a huge ratio.
    if dtype == onp.float64:
        return 1e-10
    if dtype == onp.float16:
        return 1e-3
    return 0.0 if fast else 1e-5


def _seed(shape, dtype):
    # Deterministic across processes: Python's hash() of strings/tuples is
    # salted per-process (PYTHONHASHSEED), which made the seed -- and thus the
    # test data -- vary run to run. Derive it from the raw shape ints instead.
    s = 1469598103
    for d in shape:
        s = (s * 1000003 + int(d)) & 0xffffffff
    s = (s * 1000003 + onp.dtype(dtype).itemsize) & 0xffffffff
    return s


def _rand(shape, dtype):
    onp.random.seed(_seed(shape, dtype))
    return onp.random.uniform(-1.0, 1.0, size=shape).astype(dtype)


# ---------------------------------------------------------------------------
# 1. Global reduction (scalar output)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('shape', [(1,), (7,), (1000,), (64, 64), (128, 256, 4)])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_global_sum(shape, dtype):
    a_np = _rand(shape, dtype)
    a = np.array(a_np, dtype=dtype)
    out = np.sum(a)
    ref = onp.sum(a_np.astype(onp.float64))
    assert out.shape == ()
    onp.testing.assert_allclose(out.asnumpy(), ref, rtol=_tol(dtype))


@pytest.mark.parametrize('shape', [(1,), (7,), (1000,), (64, 64), (128, 256, 4)])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_global_mean(shape, dtype):
    a_np = _rand(shape, dtype)
    a = np.array(a_np, dtype=dtype)
    out = np.mean(a)
    ref = onp.mean(a_np.astype(onp.float64))
    assert out.shape == ()
    onp.testing.assert_allclose(out.asnumpy(), ref, rtol=_tol(dtype))


# ---------------------------------------------------------------------------
# 2. Outer / leading-axis reduction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('shape,axis', [
    ((64, 128), 0),
    ((32, 16, 8), 0),
    ((32, 16, 8), (0, 1)),
])
@pytest.mark.parametrize('dtype', [onp.float16, onp.float32, onp.float64])
@pytest.mark.parametrize('keepdims', [True, False])
def test_outer_axis_sum(shape, axis, dtype, keepdims):
    a_np = _rand(shape, dtype)
    a = np.array(a_np, dtype=dtype)
    out = np.sum(a, axis=axis, keepdims=keepdims)
    ref = onp.sum(a_np.astype(onp.float64), axis=axis, keepdims=keepdims)
    assert out.shape == ref.shape
    onp.testing.assert_allclose(out.asnumpy(), ref, rtol=_tol(dtype))


@pytest.mark.parametrize('shape,axis', [
    ((64, 128), 0),
    ((32, 16, 8), 0),
    ((32, 16, 8), (0, 1)),
])
@pytest.mark.parametrize('dtype', [onp.float16, onp.float32, onp.float64])
@pytest.mark.parametrize('keepdims', [True, False])
def test_outer_axis_mean(shape, axis, dtype, keepdims):
    a_np = _rand(shape, dtype)
    a = np.array(a_np, dtype=dtype)
    out = np.mean(a, axis=axis, keepdims=keepdims)
    ref = onp.mean(a_np.astype(onp.float64), axis=axis, keepdims=keepdims)
    assert out.shape == ref.shape
    onp.testing.assert_allclose(out.asnumpy(), ref, rtol=_tol(dtype))


# ---------------------------------------------------------------------------
# 3. Mixed / other axes (fall through to the generic path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('shape,axis', [
    ((64, 128), 1),     # inner axis of 2D
    ((64, 128), -1),    # negative axis
    ((32, 16, 8), 1),   # middle axis
    ((32, 16, 8), -1),  # last axis
    ((32, 16, 8), (1, 2)),  # trailing pair
])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('keepdims', [True, False])
def test_generic_axis_sum(shape, axis, dtype, keepdims):
    a_np = _rand(shape, dtype)
    a = np.array(a_np, dtype=dtype)
    out = np.sum(a, axis=axis, keepdims=keepdims)
    ref = onp.sum(a_np.astype(onp.float64), axis=axis, keepdims=keepdims)
    assert out.shape == ref.shape
    onp.testing.assert_allclose(out.asnumpy(), ref, rtol=_tol(dtype, fast=False),
                                atol=_atol(dtype, fast=False))


@pytest.mark.parametrize('shape,axis', [
    ((64, 128), 1),
    ((64, 128), -1),
    ((32, 16, 8), 1),
    ((32, 16, 8), -1),
    ((32, 16, 8), (1, 2)),
])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('keepdims', [True, False])
def test_generic_axis_mean(shape, axis, dtype, keepdims):
    a_np = _rand(shape, dtype)
    a = np.array(a_np, dtype=dtype)
    out = np.mean(a, axis=axis, keepdims=keepdims)
    ref = onp.mean(a_np.astype(onp.float64), axis=axis, keepdims=keepdims)
    assert out.shape == ref.shape
    onp.testing.assert_allclose(out.asnumpy(), ref, rtol=_tol(dtype, fast=False),
                                atol=_atol(dtype, fast=False))


# ---------------------------------------------------------------------------
# 4. kAddTo-style accumulation via autograd
# ---------------------------------------------------------------------------
def test_global_sum_backward_ones():
    """grad of a global sum w.r.t. the input is all ones."""
    a_np = _rand((4, 5), onp.float32)
    x = np.array(a_np, dtype=onp.float32)
    x.attach_grad()
    with mx.autograd.record():
        y = np.sum(x)
    y.backward()
    onp.testing.assert_allclose(x.grad.asnumpy(),
                                onp.ones((4, 5), dtype=onp.float32),
                                rtol=1e-6)


def test_outer_axis_sum_backward_ones():
    """grad of an axis sum broadcasts ones back over the reduced axis."""
    a_np = _rand((6, 3), onp.float32)
    x = np.array(a_np, dtype=onp.float32)
    x.attach_grad()
    with mx.autograd.record():
        y = np.sum(x, axis=0)
        # contract back to a scalar so backward sees a well-defined head grad
        z = np.sum(y)
    z.backward()
    onp.testing.assert_allclose(x.grad.asnumpy(),
                                onp.ones((6, 3), dtype=onp.float32),
                                rtol=1e-6)


def test_mean_backward():
    """grad of a global mean is 1/N everywhere."""
    a_np = _rand((4, 5), onp.float32)
    x = np.array(a_np, dtype=onp.float32)
    x.attach_grad()
    with mx.autograd.record():
        y = np.mean(x)
    y.backward()
    n = a_np.size
    onp.testing.assert_allclose(x.grad.asnumpy(),
                                onp.full((4, 5), 1.0 / n, dtype=onp.float32),
                                rtol=1e-5)


def test_kaddto_two_output_accumulation():
    """A single input feeding two reductions accumulates both grads (kAddTo)."""
    a_np = _rand((4, 5), onp.float32)
    x = np.array(a_np, dtype=onp.float32)
    x.attach_grad()
    with mx.autograd.record():
        s = np.sum(x)            # grad contribution: ones
        m = np.mean(x)           # grad contribution: 1/N
        out = s + m
    out.backward()
    n = a_np.size
    expected = onp.ones((4, 5), dtype=onp.float32) + (1.0 / n)
    onp.testing.assert_allclose(x.grad.asnumpy(), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------
def test_empty_axis_tuple():
    """axis=() is a no-op reduction: output equals input."""
    a_np = _rand((3, 4), onp.float32)
    a = np.array(a_np, dtype=onp.float32)
    out = np.sum(a, axis=())
    ref = onp.sum(a_np.astype(onp.float64), axis=())
    assert out.shape == ref.shape
    onp.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5)


@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_reduce_all_axes_explicit(dtype):
    """axis=(0,1) of a 2D array equals the global reduction."""
    a_np = _rand((16, 32), dtype)
    a = np.array(a_np, dtype=dtype)
    out = np.sum(a, axis=(0, 1))
    ref = onp.sum(a_np.astype(onp.float64), axis=(0, 1))
    assert out.shape == ()
    onp.testing.assert_allclose(out.asnumpy(), ref, rtol=_tol(dtype))
    # matches the dedicated global path too
    onp.testing.assert_allclose(out.asnumpy(), np.sum(a).asnumpy(),
                                rtol=_tol(dtype))


@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_single_element_arrays(dtype):
    """Single-element arrays reduce to that element for both sum and mean."""
    a_np = _rand((1,), dtype)
    a = np.array(a_np, dtype=dtype)
    onp.testing.assert_allclose(np.sum(a).asnumpy(),
                                onp.sum(a_np.astype(onp.float64)),
                                rtol=_tol(dtype))
    onp.testing.assert_allclose(np.mean(a).asnumpy(),
                                onp.mean(a_np.astype(onp.float64)),
                                rtol=_tol(dtype))
    # 2D single element with axis reduction
    b_np = _rand((1, 1), dtype)
    b = np.array(b_np, dtype=dtype)
    onp.testing.assert_allclose(np.sum(b, axis=0).asnumpy(),
                                onp.sum(b_np.astype(onp.float64), axis=0),
                                rtol=_tol(dtype))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
