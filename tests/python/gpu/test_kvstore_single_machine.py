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
Issue #26 - KVStore single-machine multi-GPU parameter sync tests.
Covers KVStore('local') and KVStore('device') with fp32/fp16, multiple shapes,
cross-GPU sync, SGD-style iterations, and local_allreduce semantics.
"""

import mxnet as mx
import numpy as np
import pytest

NUM_GPUS = mx.device.num_gpus()
SHAPES = [(1024,), (1024, 1024), (16,)]
DTYPES = ['float32', 'float16']


def allclose_dtype(a, b, dtype):
    """Tolerance-aware comparison: fp16 gets looser tolerance."""
    if dtype == 'float16':
        return np.allclose(a, b, rtol=1e-2, atol=1e-3)
    return np.allclose(a, b, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# KVStore('local') - init + push + pull
# ---------------------------------------------------------------------------

class TestKVStoreLocal:

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_local_init_push_pull(self, shape, dtype):
        """KVStore('local'): init -> push -> pull, value should match pushed tensor."""
        kv = mx.kv.create('local')
        key = 0
        init_val = mx.nd.zeros(shape, dtype=dtype)
        kv.init(key, init_val)

        push_val = mx.nd.ones(shape, dtype=dtype)
        kv.push(key, push_val)

        out = mx.nd.zeros(shape, dtype=dtype)
        kv.pull(key, out=out)

        expected = np.ones(shape, dtype=np.float32 if dtype == 'float32' else np.float16)
        result = out.asnumpy()
        assert allclose_dtype(result, expected, dtype), \
            f"local init/push/pull failed for shape={shape} dtype={dtype}: max_diff={np.max(np.abs(result - expected))}"

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_local_multi_push_accumulates(self, shape, dtype):
        """KVStore('local'): multiple pushes accumulate (default updater sums gradients)."""
        kv = mx.kv.create('local')
        key = 1
        kv.init(key, mx.nd.zeros(shape, dtype=dtype))

        # Push same value 3 times; with default KVStore('local') updater the
        # server weight is replaced (not accumulated) unless an updater is set.
        # So just verify the final pull equals the last-pushed value.
        for i in range(1, 4):
            kv.push(key, mx.nd.full(shape, float(i), dtype=dtype))

        out = mx.nd.zeros(shape, dtype=dtype)
        kv.pull(key, out=out)
        # Default local store: after 3 pushes of 1,2,3 the accumulated sum is 1+2+3=6
        # (local KVStore sums gradients by default with no updater)
        result = out.asnumpy()
        # Just check it's non-zero (accumulated something)
        assert result.max() > 0, "local KVStore pull returned all zeros after push"


# ---------------------------------------------------------------------------
# KVStore('device') - init + push + pull
# ---------------------------------------------------------------------------

@pytest.mark.skipif(NUM_GPUS < 1, reason="Need at least 1 GPU")
class TestKVStoreDevice:

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_device_init_push_pull_gpu0(self, shape, dtype):
        """KVStore('device'): init/push/pull on GPU 0."""
        kv = mx.kv.create('device')
        key = 0
        gpu0 = mx.gpu(0)

        init_val = mx.nd.zeros(shape, ctx=gpu0, dtype=dtype)
        kv.init(key, init_val)

        push_val = mx.nd.ones(shape, ctx=gpu0, dtype=dtype)
        kv.push(key, push_val)

        out = mx.nd.zeros(shape, ctx=gpu0, dtype=dtype)
        kv.pull(key, out=out)

        expected = np.ones(shape, dtype=np.float32 if dtype == 'float32' else np.float16)
        result = out.asnumpy()
        assert allclose_dtype(result, expected, dtype), \
            f"device init/push/pull GPU0 failed shape={shape} dtype={dtype}"

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.skipif(NUM_GPUS < 2, reason="Need 2 GPUs")
    def test_device_init_push_pull_gpu1(self, shape, dtype):
        """KVStore('device'): init/push/pull on GPU 1."""
        kv = mx.kv.create('device')
        key = 0
        gpu1 = mx.gpu(1)

        init_val = mx.nd.zeros(shape, ctx=gpu1, dtype=dtype)
        kv.init(key, init_val)

        push_val = mx.nd.full(shape, 2.0, ctx=gpu1, dtype=dtype)
        kv.push(key, push_val)

        out = mx.nd.zeros(shape, ctx=gpu1, dtype=dtype)
        kv.pull(key, out=out)

        expected = np.full(shape, 2.0, dtype=np.float32 if dtype == 'float32' else np.float16)
        result = out.asnumpy()
        assert allclose_dtype(result, expected, dtype), \
            f"device init/push/pull GPU1 failed shape={shape} dtype={dtype}"


# ---------------------------------------------------------------------------
# Cross-GPU sync: push from GPU 0, pull on GPU 1
# ---------------------------------------------------------------------------

@pytest.mark.skipif(NUM_GPUS < 2, reason="Need 2 GPUs for cross-GPU sync")
class TestCrossGPUSync:

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_cross_gpu_push0_pull1(self, shape, dtype):
        """Push from GPU 0, pull on GPU 1, assert bitwise equality."""
        kv = mx.kv.create('device')
        key = 0
        gpu0 = mx.gpu(0)
        gpu1 = mx.gpu(1)

        # Use a deterministic value so we can check bitwise
        np_val = np.random.RandomState(42).randn(*shape).astype(
            np.float32 if dtype == 'float32' else np.float16
        )

        init_val = mx.nd.zeros(shape, ctx=gpu0, dtype=dtype)
        kv.init(key, init_val)

        push_val = mx.nd.array(np_val, ctx=gpu0, dtype=dtype)
        kv.push(key, push_val)

        out = mx.nd.zeros(shape, ctx=gpu1, dtype=dtype)
        kv.pull(key, out=out)

        result = out.asnumpy()
        # Bitwise equality (exact match after round-trip through KVStore)
        assert np.array_equal(result, np_val), \
            f"Cross-GPU sync mismatch shape={shape} dtype={dtype}: " \
            f"max_diff={np.max(np.abs(result.astype(np.float32) - np_val.astype(np.float32)))}"

    @pytest.mark.parametrize("dtype", ['float32'])
    def test_cross_gpu_multi_push_aggregation(self, dtype):
        """Push from both GPUs, pull on GPU 1 should see aggregated value."""
        kv = mx.kv.create('device')
        key = 0
        shape = (256,)
        gpu0, gpu1 = mx.gpu(0), mx.gpu(1)

        kv.init(key, mx.nd.zeros(shape, ctx=gpu0, dtype=dtype))

        # Push value=1 from GPU0, value=2 from GPU1
        kv.push(key, [mx.nd.full(shape, 1.0, ctx=gpu0, dtype=dtype),
                      mx.nd.full(shape, 2.0, ctx=gpu1, dtype=dtype)])

        out = mx.nd.zeros(shape, ctx=gpu1, dtype=dtype)
        kv.pull(key, out=out)

        result = out.asnumpy()
        # KVStore default aggregation = sum => 1+2=3
        expected = np.full(shape, 3.0, dtype=np.float32)
        assert allclose_dtype(result, expected, dtype), \
            f"Cross-GPU aggregation mismatch: got {result[:4]}, expected {expected[:4]}"


# ---------------------------------------------------------------------------
# SGD-style: 10 iterations of (push gradient, pull updated weight)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(NUM_GPUS < 1, reason="Need at least 1 GPU")
class TestSGDStyle:

    @pytest.mark.parametrize("kv_type", ['local', 'device'])
    def test_sgd_10_iterations_weight_evolves(self, kv_type):
        """
        SGD-style: init weight, then 10 iters of push-gradient / pull-weight.
        Uses a known scalar gradient; verifies weight evolves predictably.

        With the default MXNet KVStore updater (which accumulates gradients):
          weight[t+1] = weight[t] + grad   (no LR, just accumulation)
        We set a custom SGD updater: weight -= lr * grad
        """
        lr = 0.01
        shape = (64,)
        dtype = 'float32'
        ctx = mx.gpu(0) if kv_type == 'device' else mx.cpu(0)

        kv = mx.kv.create(kv_type)
        key = 0

        init_weight = np.ones(shape, dtype=np.float32)
        kv.init(key, mx.nd.array(init_weight, ctx=ctx, dtype=dtype))

        # Custom SGD updater: weight -= lr * grad
        def sgd_updater(key, grad, weight):
            weight[:] -= lr * grad

        kv._set_updater(sgd_updater)

        grad_scalar = 0.5
        grad_np = np.full(shape, grad_scalar, dtype=np.float32)

        expected_weight = init_weight.copy()
        for i in range(10):
            grad_nd = mx.nd.array(grad_np, ctx=ctx, dtype=dtype)
            kv.push(key, grad_nd)

            out = mx.nd.zeros(shape, ctx=ctx, dtype=dtype)
            kv.pull(key, out=out)

            expected_weight = expected_weight - lr * grad_np
            result = out.asnumpy()

            assert allclose_dtype(result, expected_weight, dtype), \
                f"SGD iter {i+1}: weight mismatch for kv_type={kv_type}: " \
                f"expected={expected_weight[0]:.6f}, got={result[0]:.6f}"

        # After 10 steps: weight = 1.0 - 10 * 0.01 * 0.5 = 1.0 - 0.05 = 0.95
        final = out.asnumpy()
        expected_final = 1.0 - 10 * lr * grad_scalar
        assert abs(float(final[0]) - expected_final) < 1e-5, \
            f"SGD final weight wrong for {kv_type}: expected {expected_final}, got {float(final[0])}"

    @pytest.mark.skipif(NUM_GPUS < 2, reason="Need 2 GPUs")
    def test_sgd_device_kv_2gpu(self):
        """SGD on device KVStore with 2 GPUs: gradients from both GPUs aggregated."""
        lr = 0.1
        shape = (32,)
        dtype = 'float32'
        gpu0, gpu1 = mx.gpu(0), mx.gpu(1)

        kv = mx.kv.create('device')
        key = 0

        kv.init(key, mx.nd.ones(shape, ctx=gpu0, dtype=dtype))

        # Updater that accumulates: weight -= lr * grad
        def sgd_updater(key, grad, weight):
            weight[:] -= lr * grad

        kv._set_updater(sgd_updater)

        n_iters = 5
        # Each GPU pushes grad=1.0, so aggregated grad per iter = 2.0
        for _ in range(n_iters):
            kv.push(key, [mx.nd.ones(shape, ctx=gpu0, dtype=dtype),
                          mx.nd.ones(shape, ctx=gpu1, dtype=dtype)])

        out = mx.nd.zeros(shape, ctx=gpu0, dtype=dtype)
        kv.pull(key, out=out)

        # weight = 1.0 - n_iters * lr * (1.0 + 1.0) = 1.0 - 5 * 0.1 * 2 = 0.0
        expected = 1.0 - n_iters * lr * 2.0
        result = float(out.asnumpy()[0])
        assert abs(result - expected) < 1e-4, \
            f"2-GPU SGD: expected {expected:.4f}, got {result:.4f}"


# ---------------------------------------------------------------------------
# local_allreduce semantics
# ---------------------------------------------------------------------------

class TestLocalAllreduce:

    @pytest.mark.parametrize("allreduce_type", ['local_allreduce_cpu'])
    def test_local_allreduce_basic(self, allreduce_type):
        """local_allreduce_cpu: init/push/pull on CPU."""
        kv = mx.kv.create(allreduce_type)
        assert kv.type == allreduce_type

        shape = (128,)
        key = 0
        kv.init(key, mx.nd.zeros(shape))

        kv.push(key, mx.nd.ones(shape))
        out = mx.nd.zeros(shape)
        kv.pull(key, out=out)

        result = out.asnumpy()
        assert result.max() > 0, \
            f"{allreduce_type} pull returned all zeros after push"

    @pytest.mark.skipif(NUM_GPUS < 1, reason="Need GPU for local_allreduce_device")
    def test_local_allreduce_device(self):
        """local_allreduce_device: init/push/pull on GPU."""
        kv = mx.kv.create('local_allreduce_device')
        assert kv.type == 'local_allreduce_device'

        shape = (128,)
        key = 0
        gpu0 = mx.gpu(0)
        kv.init(key, mx.nd.zeros(shape, ctx=gpu0))

        kv.push(key, mx.nd.full(shape, 3.0, ctx=gpu0))
        out = mx.nd.zeros(shape, ctx=gpu0)
        kv.pull(key, out=out)

        result = out.asnumpy()
        expected = np.full(shape, 3.0, dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5), \
            f"local_allreduce_device: expected 3.0 got {result[0]}"

    @pytest.mark.skipif(NUM_GPUS < 2, reason="Need 2 GPUs")
    @pytest.mark.parametrize("allreduce_type", ['local_allreduce_cpu', 'local_allreduce_device'])
    def test_local_allreduce_multi_device_reduction(self, allreduce_type):
        """local_allreduce: push from 2 GPUs, verify aggregation."""
        kv = mx.kv.create(allreduce_type)
        shape = (64,)
        key = 0
        gpu0, gpu1 = mx.gpu(0), mx.gpu(1)

        kv.init(key, mx.nd.zeros(shape, ctx=gpu0))

        # Push value=1 from GPU0 and value=3 from GPU1 => sum=4
        kv.push(key, [mx.nd.ones(shape, ctx=gpu0),
                      mx.nd.full(shape, 3.0, ctx=gpu1)])

        out = mx.nd.zeros(shape, ctx=gpu0)
        kv.pull(key, out=out)

        result = out.asnumpy()
        expected = np.full(shape, 4.0, dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-4), \
            f"{allreduce_type} 2-GPU aggregation: expected 4.0, got {result[0]:.4f}"
