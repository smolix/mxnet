"""
Stress test for apache/mxnet#19353: linalg_impl.h temp-buffer use without GPU sync.

Prior to the fix, functions like potrf/potri/gelqf/orglq/syevd/gesvd/batch_getrf/
batch_getri allocated a small scratch buffer via Storage::Get()->Alloc(), enqueued
CUDA work that used it, then called Storage::Get()->Free() from the CPU side while
the GPU kernel was still in flight.  The freed buffer could be immediately
reused by another op, causing data corruption / NaN.

The fix replaces the ad-hoc Storage::Get()->Alloc/Free pattern with a RAII wrapper
(LinalgEphemeralGPUStorage) that calls cudaStreamSynchronize() before freeing.

This test exercises several linalg operations in a tight loop interleaved with
unrelated GPU work to stress-test the synchronization.  NaN or large deviation
from a reference indicates a regression.
"""

import mxnet as mx
import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
]

NUM_ITERS = 200
DTYPE = np.float32
N = 8  # matrix size — small enough to keep the loop fast


def _make_spd_matrix(n, ctx):
    """Create a small symmetric positive-definite matrix on GPU."""
    rng = np.random.default_rng(42)
    A = rng.random((n, n)).astype(DTYPE)
    A = A @ A.T + n * np.eye(n, dtype=DTYPE)  # ensure positive-definiteness
    return mx.nd.array(A, ctx=ctx)


def _reference_cholesky(A_np):
    """NumPy reference for Cholesky."""
    return np.linalg.cholesky(A_np).astype(DTYPE)


@pytest.mark.parametrize("ctx_id", [0])
def test_potrf_no_nan_under_pressure(ctx_id):
    """
    Stress-test linalg_potrf (Cholesky) for silent NaN caused by the
    Storage::Free-before-sync race (apache/mxnet#19353).
    Runs 200 iterations of cholesky interleaved with an unrelated GPU op
    and checks the result against a NumPy reference.
    """
    ctx = mx.gpu(ctx_id)
    A_np = _make_spd_matrix(N, ctx).asnumpy()
    L_ref = _reference_cholesky(A_np)

    noise_buf = mx.nd.zeros((1024,), ctx=ctx)  # unrelated buffer to pressure allocator

    for i in range(NUM_ITERS):
        A = mx.nd.array(A_np, ctx=ctx)
        # Trigger an unrelated GPU op to make the allocator recycle buffers.
        noise_buf = noise_buf + 1.0
        # Cholesky — internally calls linalg_potrf which had the sync bug.
        L = mx.nd.linalg.potrf(A)
        mx.nd.waitall()

        L_got = L.asnumpy()
        assert not np.any(np.isnan(L_got)), (
            f"NaN detected in cholesky output at iteration {i}"
        )
        np.testing.assert_allclose(
            L_got, L_ref, rtol=1e-4, atol=1e-4,
            err_msg=f"Cholesky mismatch at iteration {i}",
        )


@pytest.mark.parametrize("ctx_id", [0])
def test_syevd_no_nan_under_pressure(ctx_id):
    """
    Stress-test linalg_syevd (symmetric eigendecomposition) for the same race.
    Uses mx.nd.linalg.syevd.
    """
    ctx = mx.gpu(ctx_id)
    A_np = _make_spd_matrix(N, ctx).asnumpy()
    # NumPy reference eigenvalues (sorted ascending, as syevd returns).
    L_ref = np.linalg.eigvalsh(A_np).astype(DTYPE)

    noise_buf = mx.nd.zeros((1024,), ctx=ctx)

    for i in range(NUM_ITERS):
        A = mx.nd.array(A_np, ctx=ctx)
        noise_buf = noise_buf + 1.0
        # syevd returns (eigenvectors, eigenvalues) in MXNet's convention
        _, L = mx.nd.linalg.syevd(A)
        mx.nd.waitall()

        L_got = L.asnumpy()
        assert not np.any(np.isnan(L_got)), (
            f"NaN detected in syevd eigenvalues at iteration {i}"
        )
        np.testing.assert_allclose(
            np.sort(L_got), np.sort(L_ref), rtol=1e-3, atol=1e-3,
            err_msg=f"syevd eigenvalue mismatch at iteration {i}",
        )


@pytest.mark.parametrize("ctx_id", [0])
def test_gelqf_no_nan_under_pressure(ctx_id):
    """
    Stress-test linalg_gelqf (LQ decomposition) for the sync race.
    """
    ctx = mx.gpu(ctx_id)
    rng = np.random.default_rng(7)
    A_np = rng.random((N, N + 2)).astype(DTYPE)  # wide matrix

    noise_buf = mx.nd.zeros((1024,), ctx=ctx)

    for i in range(NUM_ITERS):
        A = mx.nd.array(A_np, ctx=ctx)
        noise_buf = noise_buf + 1.0
        # gelqf returns (output, tau); tau contains Householder reflectors.
        out, tau = mx.nd.linalg.gelqf(A)
        mx.nd.waitall()

        out_got = out.asnumpy()
        assert not np.any(np.isnan(out_got)), (
            f"NaN detected in gelqf output at iteration {i}"
        )
        tau_got = tau.asnumpy()
        assert not np.any(np.isnan(tau_got)), (
            f"NaN detected in gelqf tau at iteration {i}"
        )


@pytest.mark.parametrize("ctx_id", [0])
def test_det_no_nan_under_pressure(ctx_id):
    """
    Stress-test linalg det (which internally calls batch_getrf + batch_getri)
    for the sync race.
    """
    ctx = mx.gpu(ctx_id)
    rng = np.random.default_rng(13)
    batch_size = 4
    A_np = rng.random((batch_size, N, N)).astype(DTYPE)
    # Make matrices non-singular.
    for b in range(batch_size):
        A_np[b] += N * np.eye(N, dtype=DTYPE)
    det_ref = np.linalg.det(A_np).astype(DTYPE)

    noise_buf = mx.nd.zeros((1024,), ctx=ctx)

    for i in range(NUM_ITERS):
        A = mx.nd.array(A_np, ctx=ctx)
        noise_buf = noise_buf + 1.0
        # mx.nd.linalg.det calls batch_getrf + diagonal product.
        sign, logdet = mx.nd.linalg.slogdet(A)
        mx.nd.waitall()

        sign_got = sign.asnumpy()
        logdet_got = logdet.asnumpy()
        det_got = sign_got * np.exp(logdet_got)

        assert not np.any(np.isnan(det_got)), (
            f"NaN detected in slogdet output at iteration {i}"
        )
        np.testing.assert_allclose(
            det_got, det_ref, rtol=1e-3, atol=1e-3,
            err_msg=f"slogdet mismatch at iteration {i}",
        )


@pytest.mark.parametrize("ctx_id", [0])
def test_mixed_linalg_pressure(ctx_id):
    """
    Run potrf + det alternately in a tight loop to maximally stress the
    Storage allocator recycling under the old (unfixed) pattern.
    """
    ctx = mx.gpu(ctx_id)
    A_np = _make_spd_matrix(N, ctx).asnumpy()
    L_ref = _reference_cholesky(A_np)

    for i in range(NUM_ITERS):
        A = mx.nd.array(A_np, ctx=ctx)
        # potrf
        L = mx.nd.linalg.potrf(A)
        # slogdet on a batch-1 array — stresses batch_getrf
        A2 = mx.nd.array(A_np[None, :, :], ctx=ctx)
        sign, logdet = mx.nd.linalg.slogdet(A2)
        mx.nd.waitall()

        L_got = L.asnumpy()
        assert not np.any(np.isnan(L_got)), (
            f"NaN in potrf at mixed iter {i}"
        )
        assert not np.any(np.isnan(logdet.asnumpy())), (
            f"NaN in slogdet at mixed iter {i}"
        )
        np.testing.assert_allclose(
            L_got, L_ref, rtol=1e-4, atol=1e-4,
            err_msg=f"potrf mismatch at mixed iter {i}",
        )


if __name__ == "__main__":
    test_potrf_no_nan_under_pressure(0)
    test_syevd_no_nan_under_pressure(0)
    test_gelqf_no_nan_under_pressure(0)
    test_det_no_nan_under_pressure(0)
    test_mixed_linalg_pressure(0)
    print("All stress tests passed.")
