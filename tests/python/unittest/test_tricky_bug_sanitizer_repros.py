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

"""Sanitizer-oriented repros for leaks that are invisible in normal runs."""

import os
import subprocess
import sys
import textwrap


def _expect_subprocess_clean_under_sanitizer(body, timeout=40):
    code = f"""
import mxnet as mx
import numpy as np

{textwrap.indent(body, '')}
"""
    env = os.environ.copy()
    env.setdefault("MXNET_ENGINE_TYPE", "NaiveEngine")
    env.setdefault("MXNET_CUDA_LIB_CHECKING", "0")
    env.setdefault("MXNET_CUDNN_LIB_CHECKING", "0")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        "subprocess was not sanitizer-clean\n"
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_np_random_laplace_ndarray_params_are_lsan_clean():
    _expect_subprocess_clean_under_sanitizer(
        """
mx.npx.set_np()
loc = mx.np.array([0.0, 1.0])
scale = mx.np.array([1.0, 2.0])
for _ in range(1000):
    mx.np.random.laplace(loc, scale, size=(2,)).wait_to_read()
"""
    )


def test_np_random_multinomial_ndarray_pvals_are_lsan_clean():
    _expect_subprocess_clean_under_sanitizer(
        """
mx.npx.set_np()
pvals = mx.np.array([0.25, 0.75])
for _ in range(1000):
    mx.np.random.multinomial(4, pvals).wait_to_read()
"""
    )


def test_autograd_function_backward_wrappers_are_lsan_clean():
    _expect_subprocess_clean_under_sanitizer(
        """
class Scale(mx.autograd.Function):
    def forward(self, x):
        return x * 3

    def backward(self, dy):
        return dy * 3

for _ in range(500):
    x = mx.nd.ones((8,), dtype='float32')
    x.attach_grad()
    with mx.autograd.record():
        y = Scale()(x).sum()
    y.backward()
mx.nd.waitall()
"""
    )


def test_multi_output_wrapper_exception_path_is_lsan_clean():
    _expect_subprocess_clean_under_sanitizer(
        """
from mxnet import _global_var

original_cls = _global_var._ndarray_cls
calls = {'count': 0}

def raising_cls(handle, stype=0):
    calls['count'] += 1
    if calls['count'] == 2:
        raise RuntimeError('stop while wrapping outputs')
    return original_cls(handle, stype=stype)

_global_var._ndarray_cls = raising_cls
try:
    try:
        mx.nd.split(mx.nd.arange(8), num_outputs=2)
    except RuntimeError:
        pass
finally:
    _global_var._ndarray_cls = original_cls
"""
    )
