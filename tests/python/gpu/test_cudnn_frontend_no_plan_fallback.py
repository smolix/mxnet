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
import subprocess
import sys
import textwrap

import pytest


def test_cudnn_frontend_no_heuristic_plans_falls_back():
    code = textwrap.dedent(
        """
        import os
        os.environ['MXNET_CUDNN_AUTOTUNE_FRONTEND'] = '1'
        os.environ['MXNET_CUDNN_ALGO_VERBOSE_LEVEL'] = '1'
        os.environ['MXNET_CUDNN_FORCE_NO_HEURISTIC_PLANS'] = '1'

        import mxnet as mx

        if mx.context.num_gpus() < 1:
            print('SKIP: requires GPU')
            raise SystemExit(42)
        if not mx.runtime.Features().is_enabled('CUDNN'):
            print('SKIP: MXNet was built without cuDNN')
            raise SystemExit(42)

        ctx = mx.gpu(0)
        mx.random.seed(1234)

        x = mx.nd.random.uniform(-1.0, 1.0, shape=(1, 5, 8, 8), ctx=ctx)
        conv_w = mx.nd.random.uniform(-1.0, 1.0, shape=(7, 5, 3, 3), ctx=ctx)
        y = mx.nd.Convolution(
            data=x,
            weight=conv_w,
            no_bias=True,
            kernel=(3, 3),
            pad=(1, 1),
            num_filter=7)
        y.wait_to_read()

        deconv_w = mx.nd.random.uniform(-1.0, 1.0, shape=(7, 5, 3, 3), ctx=ctx)
        z = mx.nd.Deconvolution(
            data=y,
            weight=deconv_w,
            no_bias=True,
            kernel=(3, 3),
            pad=(1, 1),
            num_filter=5)
        z.wait_to_read()

        print('MXNET_CHILD_OK')
        """
    )
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    result = subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        capture_output=True,
        check=False,
        timeout=300,
    )
    output = (
        result.stdout.decode(errors='replace') +
        result.stderr.decode(errors='replace')
    )
    if result.returncode == 42:
        pytest.skip(output)
    assert result.returncode == 0, output
    assert 'MXNET_CHILD_OK' in output
    assert 'Using fallback engine(s)' in output
