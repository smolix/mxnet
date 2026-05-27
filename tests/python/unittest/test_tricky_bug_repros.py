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

"""Regression repros for tricky memory-safety bugs.

Each helper subprocess expects the operation to reject invalid user input with
MXNetError. On the current buggy implementation the operation may instead
succeed, crash, or be caught by ASAN/compute-sanitizer, all of which fail the
test while keeping any memory corruption isolated from the pytest worker.
"""

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


def _expect_mxnet_error_in_subprocess(body, timeout=20):
    code = f"""
import sys
import mxnet as mx
import numpy as np

try:
{textwrap.indent(body, '    ')}
except mx.base.MXNetError:
    sys.exit(0)
except Exception as err:
    print(type(err).__name__ + ": " + str(err))
    sys.exit(3)

print("operation unexpectedly succeeded")
sys.exit(2)
"""
    env = os.environ.copy()
    env.setdefault("MXNET_ENGINE_TYPE", "NaiveEngine")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        "subprocess did not observe the expected MXNetError\n"
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _expect_success_in_subprocess(body, timeout=20):
    code = f"""
import sys
import mxnet as mx
import numpy as np

try:
{textwrap.indent(body, '    ')}
except Exception as err:
    print(type(err).__name__ + ": " + str(err))
    sys.exit(2)

sys.exit(0)
"""
    env = os.environ.copy()
    env.setdefault("MXNET_ENGINE_TYPE", "NaiveEngine")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        "subprocess did not complete successfully\n"
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_numpy_cached_op_invoke_with_out_uses_caller_output():
    _expect_success_in_subprocess(
        """
mx.npx.set_np()
x = mx.sym.var('x').as_np_ndarray()
op = mx.nd.CachedOp(x + 1)
data = mx.np.ones((2,))
out = mx.np.zeros((2,))
ret = op(data, out=out)
mx.npx.waitall()
if ret is not out:
    print("CachedOp did not return the caller-provided output")
    sys.exit(3)
np.testing.assert_allclose(out.asnumpy(), np.array([2.0, 2.0], dtype=np.float32))
"""
    )


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_contrib_index_copy_rejects_out_of_bounds_indices_cpu(bad_index):
    _expect_mxnet_error_in_subprocess(
        f"""
x = mx.nd.zeros((2, 3))
index = mx.nd.array([{bad_index}], dtype=np.int64)
new_tensor = mx.nd.ones((1, 3))
out = mx.nd.contrib.index_copy(x, index, new_tensor)
out.wait_to_read()
"""
    )


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_sparse_retain_rejects_out_of_bounds_indices_cpu(bad_index):
    _expect_mxnet_error_in_subprocess(
        f"""
dense = mx.nd.array(np.ones((2, 3), dtype=np.float32))
rsp = dense.tostype('row_sparse')
indices = mx.nd.array([{bad_index}], dtype=np.int64)
out = mx.nd.sparse.retain(rsp, indices=indices)
out.wait_to_read()
"""
    )


def test_roialign_rejects_out_of_bounds_batch_id_cpu():
    _expect_mxnet_error_in_subprocess(
        """
data = mx.nd.ones((1, 1, 4, 4))
rois = mx.nd.array(np.array([[1, 0, 0, 2, 2]], dtype=np.float32))
out = mx.nd.contrib.ROIAlign(data, rois, pooled_size=(2, 2), spatial_scale=1.0)
out.wait_to_read()
"""
    )


def test_roialign_rejects_zero_spatial_input_cpu():
    _expect_mxnet_error_in_subprocess(
        """
data = mx.nd.zeros((1, 1, 0, 1))
rois = mx.nd.array(np.array([[0, 0, 0, 0, 0]], dtype=np.float32))
out = mx.nd.contrib.ROIAlign(data, rois, pooled_size=(1, 1), spatial_scale=1.0)
out.wait_to_read()
"""
    )


def test_psroi_pooling_rejects_out_of_bounds_batch_id_cpu():
    _expect_mxnet_error_in_subprocess(
        """
data = mx.nd.ones((1, 4, 4, 4))
rois = mx.nd.array(np.array([[1, 0, 0, 2, 2]], dtype=np.float32))
out = mx.nd.contrib.PSROIPooling(
    data, rois, spatial_scale=1.0, output_dim=1, pooled_size=2, group_size=2)
out.wait_to_read()
"""
    )


def test_deformable_psroi_pooling_rejects_out_of_bounds_batch_id_cpu():
    _expect_mxnet_error_in_subprocess(
        """
data = mx.nd.ones((1, 4, 4, 4))
rois = mx.nd.array(np.array([[1, 0, 0, 2, 2]], dtype=np.float32))
trans = mx.nd.zeros((1, 2, 2, 2))
out = mx.nd.contrib.DeformablePSROIPooling(
    data, rois, trans, spatial_scale=1.0, output_dim=1, group_size=2, pooled_size=2)
out.wait_to_read()
"""
    )


def test_rroialign_rejects_out_of_bounds_batch_id_cpu():
    _expect_mxnet_error_in_subprocess(
        """
data = mx.nd.ones((1, 1, 4, 4))
rois = mx.nd.array(np.array([[1, 2, 2, 2, 2, 0]], dtype=np.float32))
out = mx.nd.contrib.RROIAlign(data, rois, pooled_size=(2, 2), spatial_scale=1.0)
out.wait_to_read()
"""
    )


@pytest.mark.parametrize("bad_index", [-1, 2])
@pytest.mark.skipif(not _gpu_available(), reason="requires CUDA")
def test_npx_index_add_rejects_out_of_bounds_indices_gpu(bad_index):
    _expect_mxnet_error_in_subprocess(
        f"""
mx.npx.set_np()
ctx = mx.gpu(0)
a = mx.np.zeros((2, 4), ctx=ctx)
ind = mx.np.array([{bad_index}], dtype='int32', ctx=ctx)
val = mx.np.ones((1, 4), ctx=ctx)
out = mx.npx.index_add(a, ind, val)
out.wait_to_read()
"""
    )


@pytest.mark.parametrize("bad_index", [-1, 2])
@pytest.mark.skipif(not _gpu_available(), reason="requires CUDA")
def test_npx_index_update_rejects_out_of_bounds_indices_gpu(bad_index):
    _expect_mxnet_error_in_subprocess(
        f"""
mx.npx.set_np()
ctx = mx.gpu(0)
a = mx.np.zeros((2, 4), ctx=ctx)
ind = mx.np.array([{bad_index}], dtype='int32', ctx=ctx)
val = mx.np.ones((1, 4), ctx=ctx)
out = mx.npx.index_update(a, ind, val)
out.wait_to_read()
"""
    )
