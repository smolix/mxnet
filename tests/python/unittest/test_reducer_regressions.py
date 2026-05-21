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


def test_bool_all_cpu_product_reducer():
    check_bool_all(mx.cpu())


def test_half_prod_nanprod_cpu_residual_init():
    check_half_prod_nanprod(mx.cpu())
