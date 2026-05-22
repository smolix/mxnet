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

import numpy as onp

import mxnet as mx
from mxnet import np as mxnp, npx


npx.set_np()


def check_bool_all(ctx):
    data = mxnp.array([[True, True, False], [True, True, True]], ctx=ctx)
    onp.testing.assert_array_equal(mxnp.all(data, axis=0).asnumpy(),
                                   onp.array([True, True, False]))
    assert bool(mxnp.all(data).asnumpy()) is False


def check_half_prod_nanprod(ctx):
    data = mx.nd.array([[1.0, 2.0], [1.0, 1.5]], ctx=ctx, dtype='float16')
    onp.testing.assert_allclose(mx.nd.prod(data, axis=1).asnumpy(),
                                onp.array([2.0, 1.5], dtype='float16'))

    with_nan = mx.nd.array([[1.0, onp.nan, 2.0], [onp.nan, onp.nan, 3.0]],
                           ctx=ctx, dtype='float16')
    onp.testing.assert_allclose(mx.nd.nanprod(with_nan, axis=1).asnumpy(),
                                onp.array([2.0, 3.0], dtype='float16'))


def check_half_bfloat_min_max(ctx):
    for dtype in ('float16', 'bfloat16'):
        neg = mx.nd.array([[-8.0, -4.0, -2.0], [-7.0, -3.0, -1.0]],
                          ctx=ctx).astype(dtype)
        pos = mx.nd.array([[8.0, 4.0, 2.0], [7.0, 3.0, 1.0]],
                          ctx=ctx).astype(dtype)

        onp.testing.assert_allclose(mx.nd.max(neg).astype('float32').asnumpy(),
                                    onp.array([-1.0], dtype='float32'))
        onp.testing.assert_allclose(mx.nd.min(pos).astype('float32').asnumpy(),
                                    onp.array([1.0], dtype='float32'))

        wide_neg = mx.nd.array(-onp.ones((64, 3), dtype='float32'),
                               ctx=ctx).astype(dtype)
        wide_pos = mx.nd.array(onp.ones((64, 3), dtype='float32'),
                               ctx=ctx).astype(dtype)
        onp.testing.assert_allclose(mx.nd.max(wide_neg, axis=1).astype('float32').asnumpy(),
                                    -onp.ones((64,), dtype='float32'))
        onp.testing.assert_allclose(mx.nd.min(wide_pos, axis=1).astype('float32').asnumpy(),
                                    onp.ones((64,), dtype='float32'))


def check_numpy_inf_min_max(ctx):
    for dtype in ('float16', 'float32', 'float64'):
        neg_inf = mxnp.array([[-onp.inf, -onp.inf]], ctx=ctx, dtype=dtype)
        pos_inf = mxnp.array([[onp.inf, onp.inf]], ctx=ctx, dtype=dtype)
        mixed_inf = mxnp.array([[-onp.inf], [onp.inf]], ctx=ctx, dtype=dtype)

        for data in (neg_inf, pos_inf, mixed_inf):
            expected = data.asnumpy()
            onp.testing.assert_array_equal(mxnp.max(data, axis=0).asnumpy(),
                                           onp.max(expected, axis=0))
            onp.testing.assert_array_equal(mxnp.min(data, axis=0).asnumpy(),
                                           onp.min(expected, axis=0))
            onp.testing.assert_array_equal(mxnp.max(data).asnumpy(), onp.max(expected))
            onp.testing.assert_array_equal(mxnp.min(data).asnumpy(), onp.min(expected))


def check_numpy_empty_reduce_out_and_shape(ctx):
    data = mxnp.empty((0, 3), ctx=ctx, dtype='float32')

    for op, expected in (
            (mxnp.sum, onp.zeros((3,), dtype='float32')),
            (mxnp.prod, onp.ones((3,), dtype='float32'))):
        out = mxnp.full((3,), -7, ctx=ctx, dtype='float32')
        ret = op(data, axis=0, out=out)
        assert ret is out
        onp.testing.assert_array_equal(out.asnumpy(), expected)

        zero_out = op(data, axis=1)
        assert zero_out.shape == (0,)
        onp.testing.assert_array_equal(zero_out.asnumpy(), onp.empty((0,), dtype='float32'))

    mean_out = mxnp.full((3,), -7, ctx=ctx, dtype='float32')
    ret = mxnp.mean(data, axis=0, out=mean_out)
    assert ret is mean_out
    onp.testing.assert_array_equal(onp.isnan(mean_out.asnumpy()), onp.ones((3,), dtype=bool))
    assert mxnp.mean(data, axis=1).shape == (0,)

    for op, expected in (
            (mxnp.any, onp.zeros((3,), dtype=bool)),
            (mxnp.all, onp.ones((3,), dtype=bool))):
        out = mxnp.empty((3,), ctx=ctx, dtype=bool)
        ret = op(data, axis=0, out=out)
        assert ret is out
        onp.testing.assert_array_equal(out.asnumpy(), expected)
        assert op(data, axis=1).shape == (0,)

    for op in (mxnp.min, mxnp.max):
        assert op(data, axis=1).shape == (0,)
        try:
            op(data, axis=0).wait_to_read()
            assert False, "{} accepted an empty reduction axis".format(op.__name__)
        except mx.base.MXNetError:
            pass


def test_bool_all_cpu_product_reducer():
    check_bool_all(mx.cpu())


def test_half_prod_nanprod_cpu_residual_init():
    check_half_prod_nanprod(mx.cpu())


def test_half_bfloat_min_max_cpu_residual_init():
    check_half_bfloat_min_max(mx.cpu())


def test_numpy_inf_min_max_cpu():
    check_numpy_inf_min_max(mx.cpu())


def test_numpy_empty_reduce_out_and_shape_cpu():
    check_numpy_empty_reduce_out_and_shape(mx.cpu())
