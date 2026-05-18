"""
Regression tests for apache/mxnet#20447:
In-place ops (+=, -=, *=, /=) must NOT change the lhs dtype.
When the rhs has a different dtype the rhs should be silently cast
to the lhs dtype (matching NumPy behaviour / array-api spec).
"""
import numpy as np
import pytest
import mxnet.numpy as mnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_inplace_dtype(op_name, lhs_dtype, rhs_dtype, lhs_val, rhs_val,
                          expected_val):
    """Run one in-place op and assert dtype preservation + correct values."""
    a = mnp.array(lhs_val, dtype=lhs_dtype)
    b = mnp.array(rhs_val, dtype=rhs_dtype)

    if op_name == 'add':
        a += b
    elif op_name == 'sub':
        a -= b
    elif op_name == 'mul':
        a *= b
    elif op_name == 'truediv':
        a /= b
    elif op_name == 'floordiv':
        a //= b
    elif op_name == 'mod':
        a %= b
    elif op_name == 'pow':
        a **= b
    else:
        raise ValueError(f'Unknown op: {op_name}')

    assert str(a.dtype) == lhs_dtype, (
        f"{op_name}: expected dtype {lhs_dtype}, got {a.dtype}")
    np.testing.assert_allclose(
        a.asnumpy(),
        np.array(expected_val, dtype=lhs_dtype),
        rtol=1e-3, atol=1e-5,
        err_msg=f"{op_name} values mismatch")


# ---------------------------------------------------------------------------
# Core tests (fp32 lhs × fp64 rhs  — was the original failure case)
# ---------------------------------------------------------------------------

class TestInplaceDtypePreservation:
    """fp32 lhs should survive all in-place ops against fp64/fp16 rhs."""

    def test_imul_fp32_fp64(self):
        """fp32 *= fp64 -- main reproducer from issue #20447."""
        a = mnp.array([1.0, 2.0, 3.0], dtype='float32')
        b = mnp.array([2.0, 3.0, 4.0], dtype='float64')
        a *= b
        assert str(a.dtype) == 'float32', f'dtype changed to {a.dtype}'
        np.testing.assert_allclose(a.asnumpy(), [2.0, 6.0, 12.0], rtol=1e-5)

    def test_imul_fp32_fp16(self):
        """fp32 *= fp16 -- lower-precision rhs also works."""
        a = mnp.array([1.0, 2.0, 3.0], dtype='float32')
        b = mnp.array([2.0, 3.0, 4.0], dtype='float16')
        a *= b
        assert str(a.dtype) == 'float32', f'dtype changed to {a.dtype}'
        np.testing.assert_allclose(a.asnumpy(), [2.0, 6.0, 12.0], rtol=1e-3)

    def test_iadd_fp32_fp64(self):
        _check_inplace_dtype('add', 'float32', 'float64',
                             [1.0, 2.0, 3.0], [10.0, 20.0, 30.0],
                             [11.0, 22.0, 33.0])

    def test_isub_fp32_fp64(self):
        _check_inplace_dtype('sub', 'float32', 'float64',
                             [10.0, 20.0, 30.0], [1.0, 2.0, 3.0],
                             [9.0, 18.0, 27.0])

    def test_itruediv_fp32_fp64(self):
        _check_inplace_dtype('truediv', 'float32', 'float64',
                             [6.0, 8.0, 9.0], [2.0, 4.0, 3.0],
                             [3.0, 2.0, 3.0])

    def test_ifloordiv_fp32_fp64(self):
        _check_inplace_dtype('floordiv', 'float32', 'float64',
                             [7.0, 8.0, 9.0], [2.0, 3.0, 4.0],
                             [3.0, 2.0, 2.0])

    def test_imod_fp32_fp64(self):
        _check_inplace_dtype('mod', 'float32', 'float64',
                             [7.0, 8.0, 9.0], [3.0, 3.0, 4.0],
                             [1.0, 2.0, 1.0])

    def test_ipow_fp32_fp64(self):
        _check_inplace_dtype('pow', 'float32', 'float64',
                             [2.0, 3.0, 4.0], [2.0, 2.0, 2.0],
                             [4.0, 9.0, 16.0])


class TestInplaceDtypeIntMixed:
    """int32 lhs × int64 rhs should stay int32."""

    def test_iadd_int32_int64(self):
        _check_inplace_dtype('add', 'int32', 'int64',
                             [1, 2, 3], [10, 20, 30],
                             [11, 22, 33])

    def test_imul_int32_int64(self):
        _check_inplace_dtype('mul', 'int32', 'int64',
                             [1, 2, 3], [4, 5, 6],
                             [4, 10, 18])

    def test_isub_int32_int64(self):
        _check_inplace_dtype('sub', 'int32', 'int64',
                             [10, 20, 30], [1, 2, 3],
                             [9, 18, 27])


class TestInplaceSameDtype:
    """Same-dtype path must still work correctly (no regression)."""

    def test_imul_fp32_fp32(self):
        a = mnp.array([1.0, 2.0, 3.0], dtype='float32')
        a *= mnp.array([2.0, 3.0, 4.0], dtype='float32')
        assert str(a.dtype) == 'float32'
        np.testing.assert_allclose(a.asnumpy(), [2.0, 6.0, 12.0])

    def test_iadd_fp64_fp64(self):
        a = mnp.array([1.0, 2.0], dtype='float64')
        a += mnp.array([5.0, 6.0], dtype='float64')
        assert str(a.dtype) == 'float64'
        np.testing.assert_allclose(a.asnumpy(), [6.0, 8.0])

    def test_imul_matches_numpy(self):
        """mxnet.numpy in-place mul result should match numpy result."""
        lhs_np = np.array([1.5, 2.5, 3.5], dtype='float32')
        rhs_np = np.array([2.0, 3.0, 4.0], dtype='float64')

        # numpy reference
        lhs_np_copy = lhs_np.copy()
        lhs_np_copy *= rhs_np

        # mxnet.numpy
        a = mnp.array([1.5, 2.5, 3.5], dtype='float32')
        b = mnp.array([2.0, 3.0, 4.0], dtype='float64')
        a *= b

        np.testing.assert_allclose(a.asnumpy(), lhs_np_copy, rtol=1e-5)
        assert str(a.dtype) == 'float32'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
