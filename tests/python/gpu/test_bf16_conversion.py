#!/usr/bin/env python3
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
bfloat16 round-trip correctness on GPU.

bf16 is the top 16 bits of an IEEE fp32 value (1 sign, 8 exponent, 7
mantissa bits) with round-to-nearest-even applied when truncating. These
tests cast float32 -> bfloat16 -> float32 on the GPU and compare against a
NumPy emulation of that rounding, and check that special values (NaN, +/-inf)
survive the round trip.
"""

import numpy as np
import pytest
import mxnet as mx
from mxnet import nd

# bf16 carries ~8 significant bits (1 implicit + 7 mantissa) -> rel err ~2^-8.
BF16_RTOL = 8e-3


def _gpu():
    try:
        a = nd.ones((1,), ctx=mx.gpu(0))
        a.wait_to_read()
    except Exception:
        pytest.skip("no usable GPU")
    return mx.gpu(0)


def _f32_to_bf16_ref(f):
    """NumPy emulation: keep the high 16 bits of fp32 with round-to-nearest-even.

    Returns the bf16-rounded value expanded back to float32.
    """
    f = np.asarray(f, dtype=np.float32)
    bits = f.view(np.uint32)
    # round-to-nearest-even: add 0x7fff plus the lsb of the surviving mantissa.
    rounding_bias = np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
    # do not perturb NaN/inf payloads; handle them separately by the caller.
    rounded = (bits.astype(np.uint64) + rounding_bias.astype(np.uint64))
    rounded = rounded.astype(np.uint32)
    truncated = (rounded >> np.uint32(16)) << np.uint32(16)
    return truncated.view(np.float32)


def _roundtrip_gpu(f32_np, ctx):
    """float32 -> bfloat16 -> float32 through the GPU."""
    x = nd.array(f32_np, ctx=ctx, dtype='float32')
    out = x.astype('bfloat16').astype('float32')
    out.wait_to_read()
    return out.asnumpy()


def test_bf16_roundtrip_known_values_gpu():
    """Known fp32 values round-trip through bf16 matching the NumPy emulation."""
    ctx = _gpu()
    vals = np.array([
        0.0, -0.0, 1.0, -1.0, 2.0, 0.5, -0.5,
        1e-3, -1e-3, 1234.5, -1234.5, 1e8, -1e8,
        3.14159265, 2.718281828,
        # values that exercise round-to-nearest-even in the mantissa
        1.0009765625, 1.0029296875, 65535.0, 0.10000000149011612,
    ], dtype=np.float32)
    got = _roundtrip_gpu(vals, ctx)
    ref = _f32_to_bf16_ref(vals)
    # exact equality with the rounding emulation (both are bf16-representable)
    np.testing.assert_array_equal(got.view(np.uint32) >> np.uint32(16),
                                  ref.view(np.uint32) >> np.uint32(16))
    # and within bf16 relative precision of the original
    nz = vals != 0
    rel = np.abs(got[nz] - vals[nz]) / np.abs(vals[nz])
    assert rel.max() < (1.0 / 128), "bf16 rel err {}".format(rel.max())


def test_bf16_roundtrip_random_gpu():
    """Random fp32 values round-trip within bf16 tolerance and match emulation."""
    ctx = _gpu()
    np.random.seed(0)
    vals = np.random.uniform(-1000, 1000, size=(256,)).astype(np.float32)
    got = _roundtrip_gpu(vals, ctx)
    ref = _f32_to_bf16_ref(vals)
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)
    np.testing.assert_allclose(got, vals, rtol=BF16_RTOL,
                               err_msg="bf16 round trip outside tolerance")


def test_bf16_elementwise_gpu():
    """bf16 elementwise ops (add, mul, relu, tanh) match fp32 within bf16 tol."""
    ctx = _gpu()
    af = nd.random.uniform(-3, 3, (128, 128), ctx=ctx)
    ab = af.astype('bfloat16')
    for name, mx_fn, np_fn in [
        ('add', lambda x: x + x, lambda x: x + x),
        ('mul', lambda x: x * 2, lambda x: x * 2),
        ('relu', lambda x: nd.relu(x), lambda x: np.maximum(x, 0)),
        ('tanh', lambda x: nd.tanh(x), np.tanh),
    ]:
        out = mx_fn(ab)
        out.wait_to_read()
        assert out.dtype == ab.dtype, name
        got = out.astype('float32').asnumpy()
        ref = np_fn(af.asnumpy())
        np.testing.assert_allclose(got, ref, rtol=BF16_RTOL, atol=BF16_RTOL,
                                   err_msg="bf16 {} mismatch".format(name))


def test_bf16_nan_inf_survive_roundtrip_gpu():
    """NaN and +/-inf survive float32 -> bf16 -> float32 on the GPU."""
    ctx = _gpu()
    vals = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    got = _roundtrip_gpu(vals, ctx)
    assert np.isnan(got[0]), "NaN did not survive bf16 round trip"
    assert np.isposinf(got[1]), "+inf did not survive bf16 round trip"
    assert np.isneginf(got[2]), "-inf did not survive bf16 round trip"


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
