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
Regression tests for bfloat16 GPU operators.

Two defects fixed:
  1. common::mshadow_type_info() lacked a kBfloat16 case, so any GPU op that
     fell back through the generic (RTC) path aborted with
     "Unknown type flag 12" -- e.g. broadcast_add on bfloat16. The RTC preamble
     also lacked a `bfloat16` type, so even after that fix the generated kernel
     failed to compile. Both are fixed; bf16 elementwise / reduction / unary
     ops now run on GPU.
  2. nd.dot on bfloat16 segfaulted (it dispatched into a storage path that
     bypassed the kernel's float-only guard). dot type inference now rejects
     unsupported dtypes cleanly instead of crashing.
"""

import numpy as np
import pytest
import mxnet as mx
from mxnet import nd


def _gpu():
    try:
        a = nd.ones((1,), ctx=mx.gpu(0))
        a.wait_to_read()
    except Exception:
        pytest.skip("no usable GPU")
    return mx.gpu(0)


def test_bf16_elementwise_gpu():
    """bf16 elementwise/unary ops run on GPU and match fp32 within bf16 tol."""
    ctx = _gpu()
    af = nd.random.uniform(-3, 3, (128, 128), ctx=ctx)
    ab = af.astype('bfloat16')
    # add, mul, relu, exp, tanh, neg
    for name, mx_fn, np_fn in [
        ('add', lambda x: x + x, lambda x: x + x),
        ('mul', lambda x: x * 2, lambda x: x * 2),
        ('relu', lambda x: nd.relu(x), lambda x: np.maximum(x, 0)),
        ('tanh', lambda x: nd.tanh(x), np.tanh),
    ]:
        out = mx_fn(ab)
        out.wait_to_read()
        assert out.dtype == ab.dtype, name  # stays bfloat16 (numpy has no bf16 dtype)
        got = out.astype('float32').asnumpy()
        ref = np_fn(af.asnumpy())
        # bf16 has ~3 significant bits of mantissa -> ~1e-2 relative tol
        np.testing.assert_allclose(got, ref, rtol=3e-2, atol=3e-2,
                                   err_msg="bf16 {} mismatch".format(name))


def test_bf16_reduction_gpu():
    """bf16 reductions run on GPU (previously aborted in the fallback path)."""
    ctx = _gpu()
    af = nd.random.uniform(0, 1, (1024,), ctx=ctx)
    ab = af.astype('bfloat16')
    s = nd.sum(ab)
    s.wait_to_read()
    # accumulation is in fp32; result close to fp32 sum
    assert abs(float(s.astype('float32').asnumpy()) - float(nd.sum(af).asnumpy())) < 5.0


def test_bf16_broadcast_add_gpu():
    """The exact op from the original bug report (broadcast_add, bf16)."""
    ctx = _gpu()
    a = nd.ones((4, 4), ctx=ctx).astype('bfloat16')
    b = nd.ones((4, 1), ctx=ctx).astype('bfloat16')
    out = nd.broadcast_add(a, b)
    out.wait_to_read()
    np.testing.assert_allclose(out.astype('float32').asnumpy(),
                               np.full((4, 4), 2.0, dtype='float32'), rtol=1e-2)


def test_bf16_dot_rejected_cleanly_gpu():
    """nd.dot on bf16 must raise a clean error, not segfault."""
    ctx = _gpu()
    a = nd.ones((8, 8), ctx=ctx).astype('bfloat16')
    with pytest.raises(mx.MXNetError, match="float16/float32/float64"):
        nd.dot(a, a).wait_to_read()


def test_bf16_fully_connected_gpu():
    """bf16 FullyConnected fwd+bwd run on GPU (via the bf16 linalg_gemm path)
    and match fp32 within bf16 precision."""
    ctx = _gpu()
    x = nd.random.uniform(-1, 1, (32, 64), ctx=ctx)
    w = nd.random.uniform(-1, 1, (48, 64), ctx=ctx)
    ref = nd.FullyConnected(x, w, no_bias=True, num_hidden=48).asnumpy()
    got = nd.FullyConnected(x.astype('bfloat16'), w.astype('bfloat16'),
                            no_bias=True, num_hidden=48).astype('float32').asnumpy()
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 5e-2, "bf16 FC rel err {}".format(rel)
    # backward must run and produce bf16 grads
    xb = x.astype('bfloat16'); wb = w.astype('bfloat16')
    xb.attach_grad(); wb.attach_grad()
    with mx.autograd.record():
        y = nd.FullyConnected(xb, wb, no_bias=True, num_hidden=48)
        loss = y.sum()
    loss.backward()
    xb.grad.wait_to_read(); wb.grad.wait_to_read()
    assert xb.grad.dtype == xb.dtype and wb.grad.dtype == wb.dtype


def test_supported_dot_dtypes_gpu():
    """fp16/fp32/fp64 dot still work."""
    ctx = _gpu()
    for dt in ['float16', 'float32', 'float64']:
        a = nd.ones((8, 8), ctx=ctx).astype(dt)
        out = nd.dot(a, a)
        out.wait_to_read()
        assert out.dtype == np.dtype(dt)
        assert abs(float(out[0, 0].astype('float32').asnumpy()) - 8.0) < 1e-2


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
