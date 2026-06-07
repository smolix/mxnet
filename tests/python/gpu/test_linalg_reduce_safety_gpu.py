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

"""GPU correctness guards for two audited bugs:

- linalg qr/solve async use-after-free: the cusolver `info`/`workspace` device
  buffers were freed on the host without a stream sync, so the still-running
  kernel could write into a block already reused by the next allocation. We
  stress this by interleaving qr/solve with large allocations that churn the
  storage pool, and check results stay correct.
- np.var/np.std/np.mean global (axis=None) reductions on GPU, which take the
  workspace-carving moments path -- a regression guard for the CUB fast-path
  workspace-aliasing hardening.
"""

import numpy as onp
import pytest
import mxnet as mx
from mxnet import np, npx

npx.set_np()


def _gpu():
    try:
        a = np.ones((1,), ctx=mx.gpu(0))
        a.wait_to_read()
    except Exception:
        pytest.skip("no usable GPU")
    return mx.gpu(0)


def test_qr_under_alloc_pressure_gpu():
    ctx = _gpu()
    onp.random.seed(1)
    for _ in range(20):
        a_np = onp.random.uniform(-1, 1, (64, 48)).astype('float32')
        a = np.array(a_np, ctx=ctx, dtype='float32')
        q, r = np.linalg.qr(a)
        # churn the storage pool so a freed-too-early buffer would be reused
        for _ in range(3):
            junk = np.zeros((256, 256), ctx=ctx, dtype='float32') + 1.0
            junk.wait_to_read()
        recon = np.matmul(q, r)
        onp.testing.assert_allclose(recon.asnumpy(), a_np, rtol=1e-3, atol=1e-3)


def test_solve_under_alloc_pressure_gpu():
    ctx = _gpu()
    onp.random.seed(2)
    for _ in range(20):
        a_np = (onp.random.uniform(-1, 1, (32, 32)) + 4 * onp.eye(32)).astype('float32')
        b_np = onp.random.uniform(-1, 1, (32, 5)).astype('float32')
        a = np.array(a_np, ctx=ctx, dtype='float32')
        b = np.array(b_np, ctx=ctx, dtype='float32')
        x = np.linalg.solve(a, b)
        for _ in range(3):
            junk = np.zeros((256, 256), ctx=ctx, dtype='float32') + 1.0
            junk.wait_to_read()
        recon = np.matmul(a, x)
        onp.testing.assert_allclose(recon.asnumpy(), b_np, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('dtype', [onp.float16, onp.float32, onp.float64])
@pytest.mark.parametrize('shape', [(64, 64), (512, 512), (1024, 1024)])
def test_global_var_std_mean_gpu(dtype, shape):
    ctx = _gpu()
    onp.random.seed(3)
    a_np = onp.random.uniform(-1, 1, shape).astype(dtype)
    a = np.array(a_np, ctx=ctx, dtype=dtype)
    ref64 = a_np.astype(onp.float64)
    tol = 2e-2 if dtype == onp.float16 else (1e-5 if dtype == onp.float32 else 1e-9)
    for op, mxf, npf in [('mean', np.mean, onp.mean),
                         ('var', np.var, onp.var),
                         ('std', np.std, onp.std)]:
        got = float(mxf(a).asnumpy())
        ref = float(npf(ref64))
        onp.testing.assert_allclose(got, ref, rtol=tol, atol=tol,
                                    err_msg="{} {} {}".format(op, dtype, shape))


@pytest.mark.parametrize('shape,axis', [((8, 200000), 1), ((200000, 8), 0),
                                        ((4, 4, 65536), 2)])
def test_fp16_mean_large_axis_no_overflow_gpu(shape, axis):
    """np.mean(fp16) over a large axis must not overflow to inf.

    The reduce writes the un-normalized sum to the fp16 output before dividing;
    for a large reduced extent that sum exceeds fp16's ~65504 range. The fix
    accumulates the sum in fp32 and divides before casting. NumPy stays finite
    here (it accumulates mean in a wider type), so MXNet must too. (np.sum on
    fp16 may still overflow -- that matches NumPy -- so we only assert on mean.)
    """
    ctx = _gpu()
    onp.random.seed(4)
    a_np = onp.random.uniform(0.5, 1.5, shape).astype('float16')
    a = np.array(a_np, ctx=ctx, dtype='float16')
    got = np.mean(a, axis=axis).asnumpy().astype('float64')
    assert not onp.isinf(got).any(), "fp16 mean overflowed to inf"
    assert not onp.isnan(got).any()
    ref = onp.mean(a_np.astype('float64'), axis=axis)
    onp.testing.assert_allclose(got, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize('shape,axis', [((8, 200000), 1), ((200000, 8), 0)])
@pytest.mark.parametrize('op', ['var', 'std'])
def test_fp16_var_std_large_axis_no_overflow_gpu(shape, axis, op):
    """np.var/np.std(fp16) over a large axis must not overflow to inf/nan.

    The moments path reduced the un-normalized sum into the fp16 output before
    dividing; for a large reduced extent that overflows fp16. The fix reduces the
    mean and the second moment into fp32 scratch and casts the (O(1)) result down
    to fp16. NumPy stays finite here, so MXNet must too.
    """
    ctx = _gpu()
    onp.random.seed(5)
    a_np = onp.random.uniform(0.5, 1.5, shape).astype('float16')
    a = np.array(a_np, ctx=ctx, dtype='float16')
    mxf = np.var if op == 'var' else np.std
    npf = onp.var if op == 'var' else onp.std
    got = mxf(a, axis=axis).asnumpy().astype('float64')
    assert not onp.isinf(got).any() and not onp.isnan(got).any(), \
        "fp16 {} overflowed".format(op)
    ref = npf(a_np.astype('float64'), axis=axis)
    onp.testing.assert_allclose(got, ref, rtol=5e-2, atol=5e-2)


def test_fp16_mean_cpu_large_axis_no_overflow():
    """np.mean(fp16) over a large axis on CPU must not overflow (companion to the
    GPU case; the CPU generic-reduce path also summed into fp16 before dividing)."""
    onp.random.seed(6)
    a_np = onp.random.uniform(0.5, 1.5, (8, 200000)).astype('float16')
    a = np.array(a_np, ctx=mx.cpu(), dtype='float16')
    got = np.mean(a, axis=1).asnumpy().astype('float64')
    assert not onp.isinf(got).any() and not onp.isnan(got).any()
    onp.testing.assert_allclose(got, onp.mean(a_np.astype('float64'), axis=1),
                                rtol=3e-2, atol=3e-2)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
