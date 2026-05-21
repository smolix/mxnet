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

"""Regression coverage for attention-shaped oneDNN batch_dot chains."""

import os

import mxnet as mx
import numpy as np
import pytest
from mxnet.test_utils import assert_almost_equal
from mxnet.util import use_np


pytestmark = pytest.mark.skipif(
    not mx.runtime.Features().is_enabled("ONEDNN")
    or os.environ.get("MXNET_ONEDNN_ENABLED") == "0",
    reason="oneDNN support is unavailable or disabled",
)


def _deterministic_array(shape, offset):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return np.sin(data * np.float32(0.17) + np.float32(offset)).astype(np.float32)


def _softmax(data, axis):
    shifted = data - np.max(data, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _numpy_batch_dot(lhs, rhs, transpose_a=False, transpose_b=False):
    if transpose_a:
        lhs = np.swapaxes(lhs, -1, -2)
    if transpose_b:
        rhs = np.swapaxes(rhs, -1, -2)
    return np.matmul(lhs, rhs)


def _mx_array(data):
    return mx.np.array(data, dtype="float32", ctx=mx.cpu())


@use_np
@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,transpose_a,transpose_b",
    [
        ((2, 3, 4), (2, 4, 5), False, False),
        ((2, 4, 3), (2, 4, 5), True, False),
        ((2, 3, 4), (2, 5, 4), False, True),
        ((2, 4, 3), (2, 5, 4), True, True),
        ((2, 3, 3, 4), (2, 3, 4, 5), False, False),
        ((2, 3, 3, 4), (2, 3, 5, 4), False, True),
    ],
)
def test_dnnl_batch_dot_default_layout_transpose_cases(
    lhs_shape, rhs_shape, transpose_a, transpose_b
):
    lhs_np = _deterministic_array(lhs_shape, 0.1)
    rhs_np = _deterministic_array(rhs_shape, 1.3)
    lhs = _mx_array(lhs_np)
    rhs = _mx_array(rhs_np)

    out = mx.npx.batch_dot(
        lhs, rhs, transpose_a=transpose_a, transpose_b=transpose_b
    )
    mx.nd.waitall()

    assert_almost_equal(
        out.asnumpy(),
        _numpy_batch_dot(lhs_np, rhs_np, transpose_a, transpose_b),
        rtol=1e-5,
        atol=1e-5,
    )


@use_np
def test_dnnl_batch_dot_attention_chain_default_layout_cpu():
    q_np = _deterministic_array((3, 5, 4), 0.1)
    k_np = _deterministic_array((3, 7, 4), 0.7)
    v_np = _deterministic_array((3, 7, 6), 1.3)
    q = _mx_array(q_np)
    k = _mx_array(k_np)
    v = _mx_array(v_np)

    scores = mx.npx.batch_dot(q, k, transpose_b=True) / np.sqrt(4.0)
    weights = mx.npx.softmax(scores, axis=2)
    out = mx.npx.batch_dot(weights, v)
    mx.nd.waitall()

    scores_np = _numpy_batch_dot(q_np, k_np, transpose_b=True) / np.sqrt(4.0)
    weights_np = _softmax(scores_np, axis=2)
    assert_almost_equal(
        out.asnumpy(),
        _numpy_batch_dot(weights_np, v_np),
        rtol=1e-5,
        atol=1e-5,
    )


@use_np
def test_dnnl_batch_dot_attention_chain_subgraph_cpu():
    class AttentionBlock(mx.gluon.HybridBlock):
        def forward(self, q, k, v):
            scores = mx.npx.batch_dot(q, k, transpose_b=True) / np.sqrt(4.0)
            weights = mx.npx.softmax(scores, axis=2)
            return mx.npx.batch_dot(weights, v)

    q_np = _deterministic_array((3, 5, 4), 0.2)
    k_np = _deterministic_array((3, 7, 4), 0.8)
    v_np = _deterministic_array((3, 7, 6), 1.4)
    q = _mx_array(q_np)
    k = _mx_array(k_np)
    v = _mx_array(v_np)

    net = AttentionBlock()
    net.initialize()
    net.hybridize()
    ref_out = net(q, k, v)

    net.optimize_for(q, k, v, backend="ONEDNN")
    out = net(q, k, v)
    mx.nd.waitall()

    assert "_sg_onednn_batch_dot" in net._cached_graph[1].tojson()
    assert_almost_equal(out.asnumpy(), ref_out.asnumpy(), rtol=1e-5, atol=1e-5)
