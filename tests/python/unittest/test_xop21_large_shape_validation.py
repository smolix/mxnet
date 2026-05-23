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

"""XOP21: regression coverage for the large-shape truncation fixes that
promote ``int`` shape counters to ``size_t`` / ``mxnet::index_t``.

These tests primarily exercise the *small* code path to ensure the type
promotions did not change numerics. The truly-large shape cases that
would actually overflow ``int`` are gated behind ``pytest.mark.skip``
because allocating ``> INT_MAX`` elements is not feasible in CI.
"""

import pytest

import mxnet as mx


def test_imports_smoke():
    """Confirm the module loads and a trivial op works."""
    x = mx.np.zeros((1,))
    y = mx.np.transpose(x)
    assert y.shape == (1,)


def test_layernorm_channel_size_one():
    """LayerNorm with channel_size == 1 (smallest case): ensures the
    int64_t channel_size promotion didn't break the trivial case."""
    layer = mx.gluon.nn.LayerNorm(axis=-1)
    layer.initialize()
    x = mx.np.ones((2, 1))  # last-axis length == 1, so channel_size == 1
    out = layer(x).asnumpy()
    # With channel_size == 1, mean == x, so normalized output is 0,
    # then scaled by gamma=1 and shifted by beta=0 -> 0.
    assert out.shape == (2, 1)


def test_layernorm_int_channel_size_within_bounds():
    """Normal-small LayerNorm exercise: covers the type-promoted
    channel_size division and reduction without hitting INT_MAX."""
    layer = mx.gluon.nn.LayerNorm(axis=-1, epsilon=1e-5)
    layer.initialize()
    x = mx.np.array(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype="float32"
    )
    out = layer(x).asnumpy()
    assert out.shape == (2, 4)
    # Each row should have ~zero mean after normalization.
    for row in out:
        assert abs(float(row.mean())) < 1e-4


def test_groupnorm_small_case():
    """GroupNorm on a small shape — covers the int64_t channel_size
    promotion in group_norm-inl.h."""
    layer = mx.gluon.nn.GroupNorm(num_groups=2, in_channels=4)
    layer.initialize()
    x = mx.np.ones((1, 4, 2, 2), dtype="float32")
    out = layer(x).asnumpy()
    assert out.shape == (1, 4, 2, 2)


@pytest.mark.skip(
    reason=(
        "Exercising the > INT_MAX element shape would require "
        "> 8 GiB allocation; the type promotion is verified at compile "
        "time and via the small-shape regression tests."
    )
)
def test_layernorm_above_int_max_elements():
    """Placeholder for the truly-large case (> INT_MAX elements)."""
    shape = (1, (1 << 31) + 16)
    layer = mx.gluon.nn.LayerNorm(axis=-1)
    layer.initialize()
    x = mx.np.zeros(shape, dtype="float32")
    layer(x).wait_to_read()


# GPU-only large-shape kernel-launch guards (XOP21 second batch).
# We can't easily exercise the > INT_MAX path without > 8 GiB allocations,
# so the regressions here just confirm the small-shape path still works
# after the guards were added.

import pytest
import mxnet as mx
import numpy as np


def _gpu_available():
    return mx.runtime.Features().is_enabled("CUDA") and mx.context.num_gpus() > 0


@pytest.mark.skipif(not _gpu_available(), reason="requires GPU for ROIAlign")
def test_roialign_small_shape_after_int_max_guard():
    """XOP21: ROIAlignForwardCompute now guards `count = out.Size()` against
    silent int truncation.  Confirm the guard does not regress the standard
    small-shape forward."""
    data = mx.nd.array(np.random.randn(1, 3, 8, 8).astype('float32'), ctx=mx.gpu(0))
    rois = mx.nd.array(np.array([[0, 0, 0, 4, 4]], dtype='float32'), ctx=mx.gpu(0))
    out = mx.nd.contrib.ROIAlign(data, rois, pooled_size=(2, 2), spatial_scale=1.0)
    assert out.shape == (1, 3, 2, 2)
    mx.nd.waitall()


@pytest.mark.skipif(not _gpu_available(), reason="requires GPU for GroupNorm CUDA path")
def test_groupnorm_small_shape_after_int_max_guard():
    """XOP21: GroupNormCompute now promotes the group element count to
    int64_t before the INT_MAX check.  Confirm the guard does not regress
    the standard small-shape forward."""
    x = mx.nd.array(np.random.randn(2, 4, 4, 4).astype('float32'), ctx=mx.gpu(0))
    gamma = mx.nd.ones((4,), ctx=mx.gpu(0))
    beta = mx.nd.zeros((4,), ctx=mx.gpu(0))
    out = mx.nd.GroupNorm(x, gamma, beta, num_groups=2)
    assert out.shape == (2, 4, 4, 4)
    mx.nd.waitall()
