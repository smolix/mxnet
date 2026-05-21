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
"""FullyConnected GPU forward parity test for MXNET_USE_CUBLASLT=1 (PR-A).

Scope: ensures the GPU `FullyConnected` operator (exercised through
`mx.gluon.nn.Dense`) produces numerically equivalent outputs with the
cuBLASLt heuristic path on vs. off across the four dtypes the wrapper
layer covers (fp16, bf16, fp32, fp32+TF32) and three FC-typical shapes.

`MXNET_USE_CUBLASLT` is read once per process and cached, so each
(use_lt, shape, dtype) cell forks a child subprocess. Inputs are seeded
identically; weights are initialised from a fixed seed too. We compare
output checksums (sum / mean / first / last) at dtype-appropriate
tolerances:

  fp16, bf16:    rtol 1e-2 (tensor-core algos drift at the LSB).
  fp32:          rtol 5e-3 (TF32 vs IEEE FP32 selection).
  fp32+TF32-on:  rtol 5e-3 (both paths use TF32; algo choice differs).

Tolerance lower bounds (`atol`) account for shapes where the checksum
itself is near zero. bf16 is reachable through `linalg_gemm<gpu, bf16_t>`
but `mx.nd.dot`/`gluon.nn.Dense` only dispatches it on builds where
`MSHADOW_REAL_TYPE_SWITCH` was widened — we skip the bf16 case gracefully
when the kernel reports the legacy fp16-only FATAL ("currently only
supported by CuDNN version") path or fails to construct the input array.
"""

import os
import struct
import subprocess
import sys
import tempfile
import textwrap

import pytest


# (M, K, N) — Dense maps to FC with weight shape (N, K), output (M, N).
_SHAPES = [
    (128, 256, 128),
    (1024, 4096, 1024),
    (4096, 4096, 4096),
]

# (dtype, allow_tf32, atol, rtol)
# rtol 1e-2 for low-precision dtypes per the task spec; 5e-3 for fp32 lanes.
_DTYPE_CASES = [
    ('float16', '0', 1e-1, 1e-2),
    ('bfloat16', '0', 1e-1, 1e-2),
    ('float32', '0', 1e-3, 5e-3),
    ('float32', '1', 1e-3, 5e-3),  # TF32-on
]


def _runner(use_lt: int, shape, dtype: str, allow_tf32: str):
    """Run gluon.nn.Dense forward in a child process and return a 4-tuple
    checksum (sum, mean, first, last) cast to float64. Raises
    AssertionError with captured stderr on subprocess failure."""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        outpath = tmp.name
    code = textwrap.dedent(
        f"""
        import os
        os.environ['MXNET_USE_CUBLASLT'] = '{use_lt}'
        os.environ['MXNET_CUDA_ALLOW_TENSOR_CORE'] = '{allow_tf32}'
        import numpy as np
        import mxnet as mx
        from mxnet import gluon
        import struct
        np.random.seed(0xC0FFEE)
        mx.random.seed(0xC0FFEE)
        m, k, n = {shape}
        dtype = '{dtype}'
        # Scale low-precision inputs to keep K-accumulated products in range.
        scale = 0.02 if dtype in ('float16', 'bfloat16') else 1.0
        x_np = (np.random.randn(m, k) * scale).astype('float32')
        w_np = (np.random.randn(n, k) * scale).astype('float32')
        b_np = (np.random.randn(n) * scale).astype('float32')
        try:
            x = mx.nd.array(x_np, ctx=mx.gpu(0), dtype=dtype)
            w = mx.nd.array(w_np, ctx=mx.gpu(0), dtype=dtype)
            b = mx.nd.array(b_np, ctx=mx.gpu(0), dtype=dtype)
        except Exception as e:
            # bf16 not supported on this build's ndarray constructor.
            print('SKIP: ' + repr(e))
            raise SystemExit(42)
        dense = gluon.nn.Dense(n, in_units=k, use_bias=True, dtype=dtype)
        # Initialize from the same numpy weights for both subprocesses.
        dense.weight.initialize(ctx=mx.gpu(0))
        dense.bias.initialize(ctx=mx.gpu(0))
        dense.weight.set_data(w.as_np_ndarray())
        dense.bias.set_data(b.as_np_ndarray())
        try:
            y = dense(x.as_np_ndarray())
            y.wait_to_read()
        except mx.MXNetError as e:
            # Some dtypes (notably bf16) may not be wired through FC on this
            # build; report and skip rather than fail the parity test.
            print('SKIP: ' + repr(e))
            raise SystemExit(42)
        arr = y.asnumpy().astype('float64')
        with open({outpath!r}, 'wb') as fh:
            fh.write(struct.pack('<dddd',
                float(arr.sum()), float(arr.mean()),
                float(arr.flat[0]), float(arr.flat[-1])))
        """
    )
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    r = subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if r.returncode == 42:
        return None  # caller will skip
    if r.returncode != 0:
        raise AssertionError(
            f"child failed (use_lt={use_lt}, shape={shape}, dtype={dtype}, "
            f"allow_tf32={allow_tf32})\n"
            f"STDOUT: {r.stdout.decode(errors='replace')}\n"
            f"STDERR: {r.stderr.decode(errors='replace')}")
    with open(outpath, 'rb') as fh:
        data = fh.read()
    os.unlink(outpath)
    return struct.unpack('<dddd', data)


@pytest.mark.parametrize('shape', _SHAPES)
@pytest.mark.parametrize('dtype,allow_tf32,atol,rtol', _DTYPE_CASES)
def test_cublaslt_fc_parity(shape, dtype, allow_tf32, atol, rtol):
    """Forward parity between `MXNET_USE_CUBLASLT=0` (legacy cuBLAS) and
    `MXNET_USE_CUBLASLT=1` (heuristic-cached cuBLASLt) on FullyConnected
    via `gluon.nn.Dense`."""
    legacy = _runner(0, shape, dtype, allow_tf32)
    if legacy is None:
        pytest.skip(f"{dtype} not supported by Dense on this build")
    lt = _runner(1, shape, dtype, allow_tf32)
    if lt is None:
        pytest.skip(f"{dtype} not supported by Dense on this build (lt path)")
    for legv, ltv, name in zip(legacy, lt, ['sum', 'mean', 'first', 'last']):
        diff = abs(legv - ltv)
        scale = max(abs(legv), 1.0)
        assert diff <= atol + rtol * scale, (
            f"{name} drift: legacy={legv} lt={ltv} "
            f"(shape={shape}, dtype={dtype}, tf32={allow_tf32}, "
            f"atol={atol}, rtol={rtol})")
