"""Regression tests for A14 / apache#20447: in-place ops must not raise
"Type inconsistent" when lhs and rhs have different dtypes; instead rhs is
pre-cast to lhs.dtype (array-API / numpy semantics).

Issue: a1 *= b1  where a1 is float32 and b1 is int32 raised a C-level
"Type inconsistent" error.

Fix: wrap_mxnp_np_ufunc_inplace in python/mxnet/numpy/multiarray.py casts
x2 to x1.dtype before forwarding to the underlying ufunc.
"""
import numpy as np
import pytest

import mxnet as mx
from mxnet import np as mnp


def _make_pair(lhs_dtype='float32', rhs_dtype='int32'):
    lhs_data = [1.0, 2.0, 3.0]
    rhs_data = [1, 2, 3]
    a = mnp.array(lhs_data, dtype=lhs_dtype)
    b = mnp.array(rhs_data, dtype=rhs_dtype)
    na = np.array(lhs_data, dtype=lhs_dtype)
    nb = np.array(rhs_data, dtype=rhs_dtype)
    return a, b, na, nb


def test_imul_mixed_dtype():
    """a *= b with float32 lhs and int32 rhs: no error, result matches numpy."""
    a, b, na, nb = _make_pair()
    a *= b
    na *= nb
    assert a.dtype == np.dtype('float32'), f"dtype changed: {a.dtype}"
    np.testing.assert_allclose(a.asnumpy(), na, rtol=1e-6)


def test_iadd_mixed_dtype():
    """a += b with float32 lhs and int32 rhs: no error, result matches numpy."""
    a, b, na, nb = _make_pair()
    a += b
    na += nb
    assert a.dtype == np.dtype('float32'), f"dtype changed: {a.dtype}"
    np.testing.assert_allclose(a.asnumpy(), na, rtol=1e-6)


def test_isub_mixed_dtype():
    """a -= b with float32 lhs and int32 rhs: no error, result matches numpy."""
    a, b, na, nb = _make_pair()
    a -= b
    na -= nb
    assert a.dtype == np.dtype('float32'), f"dtype changed: {a.dtype}"
    np.testing.assert_allclose(a.asnumpy(), na, rtol=1e-6)


def test_itruediv_mixed_dtype():
    """a /= b with float32 lhs and int32 rhs: no error, result matches numpy."""
    a, b, na, nb = _make_pair()
    a /= b
    na /= nb
    assert a.dtype == np.dtype('float32'), f"dtype changed: {a.dtype}"
    np.testing.assert_allclose(a.asnumpy(), na, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
