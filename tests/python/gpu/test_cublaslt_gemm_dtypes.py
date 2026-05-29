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
"""Numerics parity test for MXNET_USE_CUBLASLT=1 across dtypes (PR-B).

Covers fp16, fp32 and fp64 via `mx.nd.dot` (the operator that routes to
`linalg_gemm<gpu, DType>`). Each shape is run in two child subprocesses
- one with the env flag off, one with it on - because the cuBLASLt env
flag is read once per process. We compare result checksums with
dtype-appropriate tolerances:

  fp16: 5e-3 rel (Hopper-class TF/HMMA paths can drift at the LSB).
  fp32: 5e-3 rel (TF32 algorithms vs IEEE FP32).
  fp64: 1e-10 rel (bit-near-identical, no fast paths).

bf16 is reachable through the new `linalg_gemm<gpu, bf16_t>` specialization
but `mshadow::MSHADOW_REAL_TYPE_SWITCH` does not yet dispatch bf16 through
`mx.nd.dot` (master branch). The bf16 wrapper is exercised separately by
the benchmark script `bench_cublaslt_dtypes.py` via the ctypes interface,
and indirectly by ensuring the new specialization compiles.
"""

import os
import struct
import subprocess
import sys
import tempfile
import textwrap

import pytest


_SHAPES = [
    (1024, 1024, 1024),
    (4096, 4096, 4096),
]

_DTYPE_CASES = [
    # (dtype_str, abs_tol, rel_tol)
    ('float16', 1e-1, 5e-3),
    ('float32', 1e-3, 5e-3),
    ('float64', 1e-9, 1e-10),
]


def _runner(use_lt: int, shape, dtype: str):
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
        dtype = '{dtype}'
        # Scale fp16 inputs down so accumulated products do not overflow.
        scale = 0.01 if dtype == 'float16' else 1.0
        a_np = (np.random.randn(m, k) * scale).astype(dtype)
        b_np = (np.random.randn(k, n) * scale).astype(dtype)
        a = mx.nd.array(a_np, ctx=mx.gpu(0), dtype=dtype)
        b = mx.nd.array(b_np, ctx=mx.gpu(0), dtype=dtype)
        c = mx.nd.dot(a, b)
        c.wait_to_read()
        arr = c.asnumpy().astype('float64')
        with open({outpath!r}, 'wb') as fh:
            fh.write(struct.pack('<dddd',
                float(arr.sum()), float(arr.mean()),
                float(arr.flat[0]), float(arr.flat[-1])))
        """
    )
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    try:
        r = subprocess.run(
            [sys.executable, '-c', code],
            env=env,
            capture_output=True,
            check=False,
            timeout=600,
        )
        if r.returncode != 0:
            raise AssertionError(
                f"child failed (use_lt={use_lt}, shape={shape}, dtype={dtype})\n"
                f"STDOUT: {r.stdout.decode(errors='replace')}\n"
                f"STDERR: {r.stderr.decode(errors='replace')}")
        with open(outpath, 'rb') as fh:
            data = fh.read()
    finally:
        try:
            os.unlink(outpath)
        except FileNotFoundError:
            pass
    return struct.unpack('<dddd', data)


@pytest.mark.parametrize('shape', _SHAPES)
@pytest.mark.parametrize('dtype,abs_tol,rel_tol', _DTYPE_CASES)
def test_cublaslt_matches_legacy_dtype(shape, dtype, abs_tol, rel_tol):
    legacy = _runner(0, shape, dtype)
    lt     = _runner(1, shape, dtype)
    for legv, ltv, name in zip(legacy, lt,
                               ['sum', 'mean', 'first', 'last']):
        diff = abs(legv - ltv)
        scale = max(abs(legv), 1.0)
        assert diff <= abs_tol + rel_tol * scale, (
            f"{name} drift: legacy={legv} lt={ltv} "
            f"(shape={shape}, dtype={dtype})")
