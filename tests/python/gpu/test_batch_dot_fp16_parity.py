#!/usr/bin/env python3
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
Regression test for apache/mxnet#18584:
  nd.batch_dot and nd.dot must give consistent fp16 GPU results.

Before the fix, batch_dot used cublasHgemmStridedBatched (fp16 accumulator)
while dot used cublasSgemmEx (fp32 accumulator), producing materially different
answers (max relative error > 500% on typical random inputs).

After the fix, batch_dot uses cublasGemmStridedBatchedEx with
CUBLAS_COMPUTE_32F (fp32 accumulator, fp16 I/O) and the two paths agree within
fp16 rounding (max relative error < 5e-3).
"""

import numpy as np
import pytest
import mxnet as mx


REL_TOL = 5e-3  # tight fp16 tolerance


def _make_fp16_tensors(batch, m, k, n, ctx, seed=42):
    """Return (A, B) as fp16 GPU arrays with shape (batch,m,k) and (batch,k,n)."""
    rng = np.random.RandomState(seed)
    a_np = rng.randn(batch, m, k).astype(np.float16)
    b_np = rng.randn(batch, k, n).astype(np.float16)
    return (mx.nd.array(a_np, ctx=ctx, dtype=np.float16),
            mx.nd.array(b_np, ctx=ctx, dtype=np.float16))


def _manual_dot(A, B):
    """Stacked 2-D nd.dot — uses cublasSgemmEx (fp32 accum, pseudo-fp16)."""
    slices = [mx.nd.dot(A[i], B[i]) for i in range(A.shape[0])]
    return mx.nd.stack(*slices)


def _max_rel_error(x, ref):
    x_np  = x.asnumpy().astype(np.float32)
    r_np  = ref.asnumpy().astype(np.float32)
    denom = np.abs(r_np) + 1e-6
    return float((np.abs(x_np - r_np) / denom).max())


@pytest.fixture
def ctx():
    if mx.context.num_gpus() < 1:
        pytest.skip("No GPU available")
    return mx.gpu(0)


class TestBatchDotFp16Parity:
    """batch_dot must be numerically consistent with dot on fp16 inputs."""

    def test_square_batch(self, ctx):
        """Standard square batch: (8,64,64) x (8,64,64)."""
        A, B = _make_fp16_tensors(8, 64, 64, 64, ctx)
        rel = _max_rel_error(mx.nd.batch_dot(A, B), _manual_dot(A, B))
        assert rel < REL_TOL, f"max rel error {rel:.4e} >= {REL_TOL} (fp16 parity broken)"

    def test_rectangular_batch(self, ctx):
        """Non-square: (4, 32, 128) x (4, 128, 64)."""
        A, B = _make_fp16_tensors(4, 32, 128, 64, ctx)
        rel = _max_rel_error(mx.nd.batch_dot(A, B), _manual_dot(A, B))
        assert rel < REL_TOL, f"max rel error {rel:.4e} >= {REL_TOL}"

    def test_transpose_a(self, ctx):
        """transpose_a=True: A shape (4,64,32) used as (4,32,64)."""
        rng = np.random.RandomState(7)
        a_np = rng.randn(4, 64, 32).astype(np.float16)
        b_np = rng.randn(4, 64, 64).astype(np.float16)
        A = mx.nd.array(a_np, ctx=ctx, dtype=np.float16)
        B = mx.nd.array(b_np, ctx=ctx, dtype=np.float16)
        bd = mx.nd.batch_dot(A, B, transpose_a=True)
        # manual: dot(A[i].T, B[i])
        ref = mx.nd.stack(*[mx.nd.dot(A[i].T, B[i]) for i in range(4)])
        rel = _max_rel_error(bd, ref)
        assert rel < REL_TOL, f"transpose_a: max rel error {rel:.4e}"

    def test_transpose_b(self, ctx):
        """transpose_b=True: B shape (4,64,32) used as (4,32,64)."""
        rng = np.random.RandomState(13)
        a_np = rng.randn(4, 32, 64).astype(np.float16)
        b_np = rng.randn(4, 64, 64).astype(np.float16)
        A = mx.nd.array(a_np, ctx=ctx, dtype=np.float16)
        B = mx.nd.array(b_np, ctx=ctx, dtype=np.float16)
        bd = mx.nd.batch_dot(A, B, transpose_b=True)
        ref = mx.nd.stack(*[mx.nd.dot(A[i], B[i].T) for i in range(4)])
        rel = _max_rel_error(bd, ref)
        assert rel < REL_TOL, f"transpose_b: max rel error {rel:.4e}"

    def test_large_batch(self, ctx):
        """Larger batch (32, 128, 128) to stress the strided path."""
        A, B = _make_fp16_tensors(32, 128, 128, 128, ctx)
        rel = _max_rel_error(mx.nd.batch_dot(A, B), _manual_dot(A, B))
        assert rel < REL_TOL, f"large batch: max rel error {rel:.4e}"

    def test_attention_pattern(self, ctx):
        """
        Simulate QK^T as used in transformer attention:
        Q (heads*batch, seq, head_dim) x K^T (heads*batch, head_dim, seq).
        """
        n_heads, batch, seq, head_dim = 8, 2, 64, 64
        B_eff = n_heads * batch
        rng = np.random.RandomState(99)
        q_np = (rng.randn(B_eff, seq, head_dim) / np.sqrt(head_dim)).astype(np.float16)
        k_np = rng.randn(B_eff, seq, head_dim).astype(np.float16)
        Q = mx.nd.array(q_np, ctx=ctx, dtype=np.float16)
        K = mx.nd.array(k_np, ctx=ctx, dtype=np.float16)
        bd  = mx.nd.batch_dot(Q, K, transpose_b=True)
        ref = mx.nd.stack(*[mx.nd.dot(Q[i], K[i].T) for i in range(B_eff)])
        rel = _max_rel_error(bd, ref)
        assert rel < REL_TOL, f"attention QK^T: max rel error {rel:.4e}"


if __name__ == "__main__":
    # Quick standalone run
    if mx.context.num_gpus() < 1:
        print("No GPU — skipping")
    else:
        ctx = mx.gpu(0)
        cases = [
            ("square 8x64x64",    lambda: _make_fp16_tensors(8, 64, 64, 64, ctx)),
            ("rect  4x32x128x64", lambda: _make_fp16_tensors(4, 32, 128, 64, ctx)),
            ("large 32x128x128",  lambda: _make_fp16_tensors(32, 128, 128, 128, ctx)),
        ]
        for name, make in cases:
            A, B = make()
            rel = _max_rel_error(mx.nd.batch_dot(A, B), _manual_dot(A, B))
            status = "PASS" if rel < REL_TOL else "FAIL"
            print(f"  [{status}] {name}: max rel err = {rel:.4e}")
