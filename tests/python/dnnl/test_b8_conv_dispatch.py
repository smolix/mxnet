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
"""FU-3 / apache#19218 regression test.

oneDNN v3 picks ``brg_conv_fwd:avx2`` for low-IC (e.g. IC=3 RGB) Conv2d at
batch=1 on AVX2-only hosts (AMD EPYC Zen 2, no AVX-512). That kernel pads
IC up to ``simd_w=8``, wasting ~63% of every AVX2 vector op on zero lanes,
and its thread scaling on bs=1 collapses across 8 NUMA nodes -- the default
64-thread case is ~10x slower than the 1-thread case (~536 ms vs ~50 ms
on EPYC 7B12).

The fix in ``src/operator/nn/dnnl/dnnl_convolution.cc`` walks ``next_impl()``
past every "brg" candidate when:

    * host is AVX2-only (no AVX-512), and
    * batch_size <= 1, and
    * IC < 8 (simd_w on AVX2).

This test asserts the pathological config -- a Conv2D ``(1,3,224,224)`` ->
64ch 3x3 -- runs in under 100 ms with the default thread pool. Pre-fix this
takes ~500 ms; post-fix it should drop into the ~50 ms range.

CPU-only test; on AVX-512 hosts the regression doesn't trigger and the
test is skipped.
"""

import platform
import time

import numpy as np
import pytest

import mxnet as mx
from mxnet import gluon, use_np


def _host_has_avx512():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        return True  # unknown / not x86: skip the perf assertion
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
    except OSError:
        return True
    flags_lines = [ln for ln in text.splitlines() if ln.startswith("flags")]
    if not flags_lines:
        return True
    flags = set(flags_lines[0].split(":", 1)[1].split())
    return any(f.startswith("avx512") for f in flags)


# Cap to 100 ms. Pre-fix, on the EPYC 7B12 reference host, the same shape
# took ~536 ms with 64 threads; the jit:avx2 path takes ~50 ms even with
# default threading.
PATHOLOGICAL_TIME_BUDGET_MS = 100.0


def _build_conv(num_filter, kernel):
    net = gluon.nn.HybridSequential()
    net.add(gluon.nn.Conv2D(channels=num_filter, kernel_size=kernel,
                            use_bias=False, activation=None))
    net.initialize(ctx=mx.cpu())
    net.hybridize()
    return net


def _measure_conv_ms(shape, num_filter, kernel, warmup=3, iters=10):
    net = _build_conv(num_filter, kernel)
    x = mx.np.array(np.random.uniform(-1.0, 1.0, size=shape).astype("float32"),
                    ctx=mx.cpu())
    # warmup -- first call constructs and caches the oneDNN primitive.
    for _ in range(warmup):
        y = net(x)
        y.wait_to_read()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = net(x)
        y.wait_to_read()
    return ((time.perf_counter() - t0) / iters) * 1000.0


@use_np
@pytest.mark.skipif(
    _host_has_avx512(),
    reason="FU-3 regression only manifests on AVX2-only hosts (no AVX-512).",
)
def test_b8_lowic_conv_fast_path():
    """The pathological FU-3 / B8 config must dispatch off brg_conv:avx2."""
    # Don't override OMP_NUM_THREADS: the bug only shows up at default
    # thread count. If a user has explicitly capped it, the fix path is
    # still exercised but the budget is loose.
    elapsed_ms = _measure_conv_ms(
        shape=(1, 3, 224, 224),
        num_filter=64,
        kernel=(3, 3),
    )
    assert elapsed_ms < PATHOLOGICAL_TIME_BUDGET_MS, (
        f"low-IC Conv2D took {elapsed_ms:.1f} ms (>= "
        f"{PATHOLOGICAL_TIME_BUDGET_MS:.0f} ms budget). "
        f"FU-3 brg_conv:avx2 gating likely regressed -- check "
        f"src/operator/nn/dnnl/dnnl_convolution.cc."
    )


@use_np
def test_b8_lowic_conv_correctness():
    """The non-brg impl we select must still produce correct output.

    Runs on every CPU host (no AVX-512 gate); on AVX-512 hosts we just
    take the normal brg path and the result is the same.
    """
    net = _build_conv(num_filter=4, kernel=(3, 3))
    # Manually set conv weights to a known value so the output is
    # deterministic regardless of init.
    w = net[0].weight
    w.set_data(mx.np.ones(w.shape, ctx=mx.cpu()) / 27.0)

    shape = (1, 3, 8, 8)
    x_np = np.arange(np.prod(shape), dtype="float32").reshape(shape) / 192.0
    x = mx.np.array(x_np, ctx=mx.cpu())
    y = net(x).asnumpy()

    # 3x3 valid conv -> 6x6 spatial; 4 OC.
    assert y.shape == (1, 4, 6, 6), f"unexpected output shape {y.shape}"
    # All 4 filters are identical (all-ones / 27), so all OC outputs match.
    for c in range(1, 4):
        diff = float(np.abs(y[0, c] - y[0, 0]).max())
        assert diff < 1e-5, f"channel {c} differs from channel 0 by {diff}"
    # Sanity: values in a reasonable range.
    assert y[0, 0].min() > 0.0 and y[0, 0].max() < 1.0


if __name__ == "__main__":
    elapsed = _measure_conv_ms((1, 3, 224, 224), 64, (3, 3))
    print(f"Conv2D (1,3,224,224) -> 64ch 3x3 took {elapsed:.2f} ms")
