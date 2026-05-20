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

import os
import platform
import subprocess
import sys
import textwrap

import mxnet as mx
import pytest


def _is_aarch64():
    return platform.machine().lower() in ("arm64", "aarch64")


def _has_onednn():
    return mx.runtime.Features().is_enabled("ONEDNN")


@pytest.mark.skipif(not _is_aarch64(), reason="Apple Silicon/AArch64 oneDNN fallback test")
@pytest.mark.skipif(not _has_onednn(), reason="MXNet was built without oneDNN")
def test_aarch64_onednn_jit_ops_fall_back_in_fresh_process():
    code = textwrap.dedent(
        """
        import mxnet as mx

        mx.random.seed(7)
        x4 = mx.nd.random.uniform(shape=(1, 1, 8, 8))
        w4 = mx.nd.random.uniform(shape=(1, 1, 3, 3))
        x2 = mx.nd.random.uniform(shape=(3, 4))
        y2 = mx.nd.random.uniform(shape=(4, 2))
        xb = mx.nd.random.uniform(shape=(2, 3, 4))
        yb = mx.nd.random.uniform(shape=(2, 4, 5))
        np_lhs = mx.np.random.uniform(size=(2, 3, 32, 32))
        np_rhs = mx.np.random.uniform(size=(2, 3, 32, 32))

        outputs = [
            mx.nd.Activation(x4, act_type="relu"),
            mx.nd.LeakyReLU(x4, act_type="leaky", slope=0.1),
            mx.nd.Pooling(x4, kernel=(2, 2), stride=(2, 2), pool_type="max"),
            mx.nd.Pooling(x4, kernel=(2, 2), stride=(2, 2), pool_type="avg"),
            mx.nd.Convolution(x4, w4, no_bias=True, kernel=(3, 3), num_filter=1),
            mx.nd.Deconvolution(x4, w4, no_bias=True, kernel=(3, 3), num_filter=1),
            mx.nd.softmax(x2, axis=1),
            mx.nd.log_softmax(x2, axis=1),
            mx.nd.dot(x2, y2),
            mx.nd.batch_dot(xb, yb),
            np_lhs + np_rhs,
        ]

        gamma = mx.nd.ones((1,))
        beta = mx.nd.zeros((1,))
        moving_mean = mx.nd.zeros((1,))
        moving_var = mx.nd.ones((1,))
        outputs.append(
            mx.nd.BatchNorm(
                x4,
                gamma,
                beta,
                moving_mean,
                moving_var,
                fix_gamma=False,
                use_global_stats=True,
            )
        )

        for out in outputs:
            if isinstance(out, (list, tuple)):
                for item in out:
                    item.wait_to_read()
            else:
                out.wait_to_read()
        print("ok")
        """
    )
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout
