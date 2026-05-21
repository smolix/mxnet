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


@use_np
def test_dnnl_batch_dot_attention_chain_default_layout_cpu():
    q = mx.np.ones((16, 100, 3), dtype="float32", ctx=mx.cpu())
    k = mx.np.ones((16, 100, 3), dtype="float32", ctx=mx.cpu())
    v = mx.np.ones((16, 100, 3), dtype="float32", ctx=mx.cpu())

    scores = mx.npx.batch_dot(q, k, transpose_b=True) / np.sqrt(3.0)
    weights = mx.npx.softmax(scores, axis=2)
    out = mx.npx.batch_dot(weights, v)
    mx.nd.waitall()

    out_np = out.asnumpy()
    assert np.isfinite(out_np).all()
    assert_almost_equal(out_np, np.ones((16, 100, 3), dtype=np.float32))
