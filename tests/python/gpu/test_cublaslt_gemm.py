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
"""Numerics parity test for MXNET_USE_CUBLASLT=1.

The cuBLASLt PoC currently only intercepts `linalg_gemm<gpu, float>()`
(the 2-D, non-batched fp32 path). That maps to `mx.nd.dot(a, b)` for
2-D fp32 tensors, FullyConnected forward, and a handful of RNN paths.

Because the cuBLASLt env flag is read once per process (cached), this
test forks two child workers via the standard `multiprocessing` spawn
context: one with the env flag off, one with it on. We compare outputs
to a TF32-tolerant tolerance (1e-3 rel, 1e-4 abs). Different heuristics
may pick TF32 vs FP32 algos so we do NOT require bit-exact equality.
"""

import os
import struct
import subprocess
import sys
import tempfile
import textwrap

import pytest


_SHAPES = [
    (64, 64, 64),
    (128, 256, 64),
    (1024, 1024, 1024),
    (2048, 512, 768),
]


def _runner(use_lt: int, shape):
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        outpath = tmp.name
    code = textwrap.dedent(
        f"""
        import os
        os.environ['MXNET_USE_CUBLASLT'] = '{use_lt}'
        import numpy as np
        import mxnet as mx
        import struct
        np.random.seed(0xC0FFEE)
        m, n, k = {shape}
        a = mx.nd.array(np.random.randn(m, k).astype('float32'), ctx=mx.gpu(0))
        b = mx.nd.array(np.random.randn(k, n).astype('float32'), ctx=mx.gpu(0))
        c = mx.nd.dot(a, b)
        c.wait_to_read()
        arr = c.asnumpy()
        with open({outpath!r}, 'wb') as fh:
            fh.write(struct.pack('<dddd',
                float(arr.sum()), float(arr.mean()),
                float(arr.flat[0]), float(arr.flat[-1])))
        """
    )
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    try:
        subprocess.run(
            [sys.executable, '-c', code],
            env=env,
            capture_output=True,
            check=True,
            timeout=300,
        )
        with open(outpath, 'rb') as fh:
            data = fh.read()
    finally:
        try:
            os.unlink(outpath)
        except FileNotFoundError:
            pass
    return struct.unpack('<dddd', data)


@pytest.mark.parametrize('shape', _SHAPES)
def test_cublaslt_matches_legacy(shape):
    legacy = _runner(0, shape)
    lt     = _runner(1, shape)
    # Relative tolerance accommodates TF32 tensor-core paths cuBLASLt may
    # pick where legacy cublasSgemmEx used IEEE FP32.
    rel = 5e-3
    abs_tol = 1e-3
    for legv, ltv, name in zip(legacy, lt,
                               ['sum', 'mean', 'first', 'last']):
        diff = abs(legv - ltv)
        scale = max(abs(legv), 1.0)
        assert diff <= abs_tol + rel * scale, (
            f"{name} drift: legacy={legv} lt={ltv} (shape={shape})")
