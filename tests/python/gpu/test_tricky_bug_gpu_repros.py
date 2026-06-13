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

"""GPU repros for tricky CUDA memory and concurrency bugs."""

import os
import subprocess
import sys
import textwrap

import pytest

import mxnet as mx


def _gpu_available():
    if not mx.runtime.Features().is_enabled("CUDA"):
        return False
    try:
        return mx.context.num_gpus() > 0
    except mx.base.MXNetError:
        return False


def _run_gpu_subprocess(body, timeout=60, extra_env=None):
    code = f"""
import ctypes
import gc
import sys
import threading
import mxnet as mx
import numpy as np
from mxnet.base import _LIB, check_call

{textwrap.indent(body, '')}
"""
    env = os.environ.copy()
    env.setdefault("MXNET_ENGINE_TYPE", "NaiveEngine")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def _expect_gpu_success(body, timeout=60, extra_env=None):
    result = _run_gpu_subprocess(body, timeout=timeout, extra_env=extra_env)
    assert result.returncode == 0, (
        "GPU subprocess did not complete successfully\n"
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _expect_gpu_mxnet_error(body, timeout=60):
    result = _run_gpu_subprocess(
        f"""
try:
{textwrap.indent(body, '    ')}
except mx.base.MXNetError:
    sys.exit(0)
except Exception as err:
    print(type(err).__name__ + ": " + str(err))
    sys.exit(3)
print("operation unexpectedly succeeded")
sys.exit(2)
""",
        timeout=timeout,
    )
    assert result.returncode == 0, (
        "GPU subprocess did not observe the expected MXNetError\n"
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="requires CUDA")


def test_mx_push_stream_dep_does_not_use_freed_ndarray_handle():
    # 500 iterations (~17s) is enough to exercise the freed-handle path while
    # staying within the 60s subprocess timeout; the per-iteration gc.collect()
    # makes 2000 iterations exceed it without testing anything new.
    _expect_gpu_success(
        """
for _ in range(500):
    arr = mx.nd.ones((1,), ctx=mx.gpu(0))
    check_call(_LIB.MXPushStreamDepEx(arr.handle, ctypes.c_size_t(0)))
    del arr
    gc.collect()
mx.nd.waitall()
"""
    )


def test_mx_get_current_stream_is_lsan_clean():
    _expect_gpu_success(
        """
for _ in range(20000):
    stream = ctypes.c_size_t()
    check_call(_LIB.MXGetCurrentStreamEx(ctypes.c_int(0), ctypes.byref(stream)))
"""
    )


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_contrib_index_copy_rejects_out_of_bounds_indices_gpu(bad_index):
    _expect_gpu_mxnet_error(
        f"""
ctx = mx.gpu(0)
x = mx.nd.zeros((2, 3), ctx=ctx)
index = mx.nd.array([{bad_index}], dtype=np.int64, ctx=ctx)
new_tensor = mx.nd.ones((1, 3), ctx=ctx)
mx.nd.contrib.index_copy(x, index, new_tensor).wait_to_read()
"""
    )


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_sparse_retain_rejects_out_of_bounds_indices_gpu(bad_index):
    _expect_gpu_mxnet_error(
        f"""
ctx = mx.gpu(0)
dense = mx.nd.array(np.ones((2, 3), dtype=np.float32), ctx=ctx)
rsp = dense.tostype('row_sparse')
indices = mx.nd.array([{bad_index}], dtype=np.int64, ctx=ctx)
mx.nd.sparse.retain(rsp, indices=indices).wait_to_read()
"""
    )


def test_roialign_rejects_bad_gpu_roi_batch_id():
    _expect_gpu_mxnet_error(
        """
ctx = mx.gpu(0)
data = mx.nd.ones((1, 1, 4, 4), ctx=ctx)
rois = mx.nd.array(np.array([[1, 0, 0, 2, 2]], dtype=np.float32), ctx=ctx)
mx.nd.contrib.ROIAlign(data, rois, pooled_size=(2, 2), spatial_scale=1.0).wait_to_read()
"""
    )


def test_roialign_rejects_zero_spatial_input_gpu():
    _expect_gpu_mxnet_error(
        """
ctx = mx.gpu(0)
data = mx.nd.zeros((1, 1, 0, 1), ctx=ctx)
rois = mx.nd.array(np.array([[0, 0, 0, 0, 0]], dtype=np.float32), ctx=ctx)
mx.nd.contrib.ROIAlign(data, rois, pooled_size=(1, 1), spatial_scale=1.0).wait_to_read()
"""
    )


def test_psroi_pooling_rejects_bad_gpu_roi_batch_id():
    _expect_gpu_mxnet_error(
        """
ctx = mx.gpu(0)
data = mx.nd.ones((1, 4, 4, 4), ctx=ctx)
rois = mx.nd.array(np.array([[1, 0, 0, 2, 2]], dtype=np.float32), ctx=ctx)
mx.nd.contrib.PSROIPooling(
    data, rois, spatial_scale=1.0, output_dim=1, pooled_size=2, group_size=2).wait_to_read()
"""
    )


def test_deformable_psroi_pooling_rejects_bad_gpu_roi_batch_id():
    _expect_gpu_mxnet_error(
        """
ctx = mx.gpu(0)
data = mx.nd.ones((1, 4, 4, 4), ctx=ctx)
rois = mx.nd.array(np.array([[1, 0, 0, 2, 2]], dtype=np.float32), ctx=ctx)
trans = mx.nd.zeros((1, 2, 2, 2), ctx=ctx)
mx.nd.contrib.DeformablePSROIPooling(
    data, rois, trans, spatial_scale=1.0, output_dim=1,
    group_size=2, pooled_size=2).wait_to_read()
"""
    )


def test_multi_lamb_lans_gpu_host_allocations_are_lsan_clean():
    _expect_gpu_success(
        """
ctx = mx.gpu(0)
weights = [mx.nd.ones((16,), ctx=ctx) for _ in range(2)]
grads = [mx.nd.ones((16,), ctx=ctx) for _ in range(2)]
mean = [mx.nd.zeros((16,), ctx=ctx) for _ in range(2)]
var = [mx.nd.zeros((16,), ctx=ctx) for _ in range(2)]
for _ in range(200):
    mx.nd.contrib.multi_lamb_update(
        weights, grads, mean, var, [1, 1], [0.01, 0.01], [0.0, 0.0], out=weights)
    mx.nd.contrib.multi_lans_update(
        weights, grads, mean, var, [1, 1], [0.01, 0.01], [0.0, 0.0], out=weights)
mx.nd.waitall()
"""
    )


@pytest.mark.skipif(
    os.environ.get("MXNET_TEST_CUBLASLT_STRESS") != "1",
    reason="set MXNET_TEST_CUBLASLT_STRESS=1 for the cuBLASLt workspace stress repro",
)
def test_cublaslt_shared_workspace_concurrent_gemm_is_sanitizer_clean():
    _expect_gpu_success(
        """
ctx = mx.gpu(0)
errors = []

def worker(seed):
    for i in range(50):
        lhs = mx.nd.ones((1024, 1024), ctx=ctx) * (seed + 1)
        rhs = mx.nd.ones((1024, 1024), ctx=ctx) * 0.5
        out = mx.nd.dot(lhs, rhs)
        value = out[0, 0].asscalar()
        expected = 1024 * (seed + 1) * 0.5
        if abs(value - expected) > 1e-2:
            errors.append((value, expected))

threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
if errors:
    print(errors[:3])
    sys.exit(2)
""",
        timeout=180,
        extra_env={"MXNET_USE_CUBLASLT": "1", "MXNET_ENGINE_TYPE": "ThreadedEnginePerDevice"},
    )


@pytest.mark.skipif(
    os.environ.get("MXNET_TEST_CUDNN_CACHE_STRESS") != "1",
    reason="set MXNET_TEST_CUDNN_CACHE_STRESS=1 for the cuDNN cache stress repro",
)
def test_cudnn_op_cache_concurrent_insert_is_sanitizer_clean():
    _expect_gpu_success(
        """
ctx = mx.gpu(0)

def worker(offset):
    for i in range(80):
        channels = 1 + ((i + offset) % 8)
        width = 8 + ((i + offset) % 13)
        data = mx.nd.ones((1, channels, width, width), ctx=ctx)
        weight = mx.nd.ones((channels, channels, 3, 3), ctx=ctx)
        mx.nd.Convolution(data, weight, num_filter=channels, kernel=(3, 3), pad=(1, 1)).wait_to_read()

threads = [threading.Thread(target=worker, args=(i * 17,)) for i in range(8)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
""",
        timeout=180,
        extra_env={
            "MXNET_CUDNN_AUTOTUNE_FRONTEND": "1",
            "MXNET_ENGINE_TYPE": "ThreadedEnginePerDevice",
        },
    )
