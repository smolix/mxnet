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

"""Regression tests for PR #15/#23: TF32 enable in FP32 transposed convolution.

Before the fix, cudnn_deconvolution-inl.h was missing the
  GetEnvAllowTensorCore() + GetEnvAllowTensorCoreConversion()
block that cudnn_convolution-inl.h already had.  On sm_80+ hardware this meant
FP32 deconvolution silently ran in full FP32 accumulation mode (~2x slower)
even though TF32 was available.

Tests here verify:
  1. Correctness: output is finite and has the correct shape.
  2. Performance heuristic: 20 iterations of a typical segmentation-decoder
     deconv complete well within the TF32-enabled time budget on sm_80+.
"""

import time
import pytest
import numpy as np
import mxnet as mx
from mxnet import np as mnp, npx, gluon
from mxnet.util import get_cuda_compute_capability

npx.set_np()

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_gpu = pytest.mark.skipif(
    mx.context.num_gpus() == 0,
    reason="requires at least one GPU",
)

requires_sm80 = pytest.mark.skipif(
    mx.context.num_gpus() == 0
    or get_cuda_compute_capability(mx.gpu(0)) < 80,
    reason="TF32 requires sm_80+",
)


# ---------------------------------------------------------------------------
# 1. Correctness: output shape + finite values (runs on any GPU)
# ---------------------------------------------------------------------------

@requires_gpu
def test_deconv_correctness_vs_expected_shape():
    """FP32 deconv output must be finite and have the correct spatial shape.

    Conv2DTranspose with stride=2 doubles each spatial dimension.
    """
    ctx = mx.gpu(0)
    layer = gluon.nn.Conv2DTranspose(
        channels=64, kernel_size=3, padding=1, strides=2,
        in_channels=64, use_bias=False,
    )
    layer.initialize(ctx=ctx)

    x = mnp.random.uniform(size=(4, 64, 14, 14), ctx=ctx).astype('float32')
    y = layer(x)
    y.wait_to_read()

    # stride=2 + padding=1 + kernel=3: output = (14-1)*2 - 2*1 + 3 = 27
    assert y.shape == (4, 64, 27, 27), f"Unexpected output shape: {y.shape}"
    arr = y.asnumpy()
    assert np.all(np.isfinite(arr)), "Non-finite values in deconv output"


@requires_gpu
def test_deconv_correctness_no_bias():
    """Zero-weight deconv should produce all-zero output."""
    ctx = mx.gpu(0)
    layer = gluon.nn.Conv2DTranspose(
        channels=16, kernel_size=3, padding=1, strides=1,
        in_channels=16, use_bias=False,
    )
    layer.initialize(ctx=ctx)

    x = mnp.ones((2, 16, 8, 8), ctx=ctx, dtype='float32')
    # Run one forward pass to materialise deferred parameter allocation.
    _ = layer(x)
    _.wait_to_read()

    # Force weights to zero, then re-run.
    layer.weight.set_data(mnp.zeros_like(layer.weight.data()))
    y = layer(x)
    y.wait_to_read()

    np.testing.assert_allclose(
        y.asnumpy(), 0.0, atol=1e-6,
        err_msg="Expected all-zero output from zero-weight deconv",
    )


# ---------------------------------------------------------------------------
# 2. Performance heuristic: TF32 must be fast enough on sm_80+
# ---------------------------------------------------------------------------

@requires_sm80
def test_deconv_fp32_tf32_enabled():
    """Verify FP32 deconv meets TF32 speed expectations on sm_80+.

    A 256-channel 3x3 deconv on a 28x28 feature map is a canonical segmentation
    decoder shape.  With TF32 on cuDNN 9 / sm_80+, 20 iterations should
    complete well under 500 ms.  Without TF32 (pre-fix), the same workload on
    Blackwell takes ~2x longer; using 500 ms as the bound gives comfortable
    margin while still catching a regression.
    """
    ctx = mx.gpu(0)
    layer = gluon.nn.Conv2DTranspose(
        channels=256, kernel_size=3, padding=1, strides=2,
        in_channels=256, use_bias=False,
    )
    layer.initialize(ctx=ctx)

    x = mnp.random.uniform(size=(16, 256, 28, 28), ctx=ctx).astype('float32')

    # Warmup — ensure cuDNN algorithm selection is done before timing.
    for _ in range(3):
        y = layer(x)
        y.wait_to_read()

    # Timed loop.
    t0 = time.time()
    for _ in range(20):
        y = layer(x)
        y.wait_to_read()
    elapsed = time.time() - t0

    assert elapsed < 0.5, (
        f"Deconv TF32 too slow: {elapsed * 1000:.1f} ms for 20 iters "
        f"(expected < 500 ms with TF32 enabled on sm_80+). "
        f"TF32 may not be enabled — check PR #15 fix in cudnn_deconvolution-inl.h."
    )


@requires_sm80
def test_deconv_fp32_output_finite_under_tf32():
    """With TF32 enabled on sm_80+, output must still be finite (no overflow)."""
    ctx = mx.gpu(0)
    layer = gluon.nn.Conv2DTranspose(
        channels=128, kernel_size=5, padding=2, strides=2,
        in_channels=128, use_bias=True,
    )
    layer.initialize(ctx=ctx)

    x = mnp.random.normal(0, 1, (8, 128, 16, 16)).astype('float32').as_in_context(ctx)
    y = layer(x)
    y.wait_to_read()

    arr = y.asnumpy()
    assert np.all(np.isfinite(arr)), (
        "Non-finite values in TF32 deconv output — possible precision overflow"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
