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


def test_bool_all_gpu_product_reducer():
    check_bool_all(mx.gpu(0))


def test_half_prod_nanprod_gpu_residual_init():
    check_half_prod_nanprod(mx.gpu(0))


def test_layer_norm_fp16_non_last_axis_large_reduction_moments():
    ctx = mx.gpu(0)
    data_np = (onp.arange(70000 * 2, dtype='float32').reshape(70000, 2) % 7) - 3
    gamma_np = onp.linspace(0.5, 1.5, 70000, dtype='float32')
    beta_np = onp.linspace(-0.25, 0.25, 70000, dtype='float32')
    data = mx.nd.array(data_np, ctx=ctx, dtype='float16')
    gamma = mx.nd.array(gamma_np, ctx=ctx, dtype='float16')
    beta = mx.nd.array(beta_np, ctx=ctx, dtype='float16')

    out, mean, std = mx.nd.LayerNorm(data, gamma, beta, axis=0, output_mean_var=True)

    expected_mean = data_np.mean(axis=0, keepdims=True)
    expected_std = onp.sqrt(((data_np - expected_mean) ** 2).mean(axis=0, keepdims=True) + 1e-5)
    expected_out = ((data_np - expected_mean) / expected_std) * gamma_np.reshape(-1, 1) + \
        beta_np.reshape(-1, 1)

    onp.testing.assert_allclose(mean.asnumpy().astype('float32'), expected_mean, rtol=2e-3, atol=2e-3)
    onp.testing.assert_allclose(std.asnumpy().astype('float32'), expected_std, rtol=2e-3, atol=2e-3)
    onp.testing.assert_allclose(out.asnumpy().astype('float32'), expected_out, rtol=2e-3, atol=2e-3)


def test_group_norm_fp16_large_reduction_moments():
    ctx = mx.gpu(0)
    data_np = (onp.arange(1 * 2 * 256 * 256, dtype='float32').reshape(1, 2, 256, 256) % 5) - 2
    gamma_np = onp.array([0.75, 1.25], dtype='float32')
    beta_np = onp.array([-0.5, 0.5], dtype='float32')
    data = mx.nd.array(data_np, ctx=ctx, dtype='float16')
    gamma = mx.nd.array(gamma_np, ctx=ctx, dtype='float16')
    beta = mx.nd.array(beta_np, ctx=ctx, dtype='float16')

    out, mean, std = mx.nd.GroupNorm(data, gamma, beta, num_groups=1, output_mean_var=True)

    expected_mean = data_np.mean(axis=(1, 2, 3), keepdims=False).reshape(1, 1)
    expected_std = onp.sqrt(((data_np - expected_mean.reshape(1, 1, 1, 1)) ** 2).mean(
        axis=(1, 2, 3), keepdims=False).reshape(1, 1) + 1e-5)
    expected_out = ((data_np - expected_mean.reshape(1, 1, 1, 1)) /
                    expected_std.reshape(1, 1, 1, 1)) * gamma_np.reshape(1, 2, 1, 1) + \
        beta_np.reshape(1, 2, 1, 1)

    onp.testing.assert_allclose(mean.asnumpy().astype('float32'), expected_mean, rtol=2e-3, atol=2e-3)
    onp.testing.assert_allclose(std.asnumpy().astype('float32'), expected_std, rtol=2e-3, atol=2e-3)
    onp.testing.assert_allclose(out.asnumpy().astype('float32'), expected_out, rtol=2e-3, atol=2e-3)
