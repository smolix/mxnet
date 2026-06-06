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

"""Type functions for the numpy module."""

from typing import NamedTuple

import numpy as onp
from .multiarray import ndarray
from .utils import _type_promotion_table


__all__ = ['can_cast', 'finfo', 'iinfo', 'result_type']

class finfo_obj(NamedTuple):
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float


class iinfo_obj(NamedTuple):
    bits: int
    max: int
    min: int


def can_cast(from_, to):
    """
    Returns True if cast between data types can occur according to
    the casting rule. If from is a scalar or array scalar,
    also returns True if the scalar value can be cast without
    overflow or truncation to an integer.
    Parameters
    ----------
    from_ : dtype, ndarray or scalar
        Data type, scalar, or array to cast from.
    to : dtype
        Data type to cast to.
    Returns
    -------
    out : bool
        True if cast can occur according to the casting rule.
    """
    if isinstance(from_, ndarray):
        from_ = from_.asnumpy()
    return onp.can_cast(from_, to)


def finfo(dtype):
    """
    Machine limits for floating-point data types.
    Notes
    -----
    `finfo` is a standard API in
    https://data-apis.org/array-api/latest/API_specification/data_type_functions.html#finfo-type
    instead of an official NumPy operator.
    Parameters
    ----------
    dtype : ndarray, float or dtype
        Kind of floating point data-type about which to get information.
    Returns
    -------
    out : finfo object
        an object having the following attributes:
            - bits : int
                number of bits occupied by the floating-point data type.
            - eps : float
                difference between 1.0 and the next smallest representable floating-point
                number larger than 1.0 according to the IEEE-754 standard.
            - max : float
                largest representable number.
            - min : float
                smallest representable number.
            - smallest_normal : float
                smallest positive floating-point number with full precision.
    """
    f_info = onp.finfo(dtype)
    return finfo_obj(f_info.bits, float(f_info.eps),
                     float(f_info.max), float(f_info.min), float(f_info.tiny))


def iinfo(dtype):
    """
    Machine limits for floating-point data types.
    Notes
    -----
    `iinfo` is a standard API in
    https://data-apis.org/array-api/latest/API_specification/data_type_functions.html#iinfo-type
    instead of an official NumPy operator.
    Parameters
    ----------
    dtype : ndarray, integer or dtype
        The kind of integer data type to get information about.
    Returns
    -------
    out : iinfo object
        an object having the following attributes:
            - bits : int
                number of bits occupied by the type
            - max : int
                largest representable number.
            - min : int
                smallest representable number.
    """
    i_info = onp.iinfo(dtype)
    return iinfo_obj(i_info.bits, i_info.max, i_info.min)


def _get_dtype(array_or_dtype):
    """Utility function for result_type"""
    if isinstance(array_or_dtype, (ndarray, onp.ndarray, onp.generic)):
        # ndarrays and NumPy scalars (e.g. np.int8(1)) carry a concrete dtype.
        return onp.dtype(array_or_dtype.dtype)
    elif isinstance(array_or_dtype, onp.dtype):
        return array_or_dtype
    else:
        try:
            return onp.dtype(array_or_dtype)
        except TypeError as err:
            raise ValueError("Inputs of result_type must be ndarrays or dtypes") from err


def _is_scalar(array_or_dtype):
    return onp.isscalar(array_or_dtype)


def _is_weak_scalar(array_or_dtype):
    """A Python builtin number is a "weak" scalar (PyTorch wrapped number): it
    contributes only its category (bool < int < float < complex), not a width.
    NumPy scalars carry a concrete dtype and are treated as strong operands."""
    return isinstance(array_or_dtype, (bool, int, float, complex)) \
        and not isinstance(array_or_dtype, onp.generic)


def _weak_category(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return 1
    if isinstance(value, float):
        return 2
    return 3


def _category_of_dtype(dt):
    dt = onp.dtype(dt)
    if dt == onp.dtype(bool):
        return 0
    if onp.issubdtype(dt, onp.integer):
        return 1
    if onp.issubdtype(dt, onp.floating):
        return 2
    return 3


def _default_dtype_for_category(cat):
    # Default per-category dtype used when a weak scalar bumps the category.
    return (onp.dtype(bool), onp.dtype('int64'),
            onp.dtype('float32'), onp.dtype('complex64'))[cat]


def _result_type_with_weak_scalars(arrays_and_dtypes):
    """PyTorch-style promotion when Python scalars are present: weak scalars only
    raise the result category; the width comes from the array/dtype operands (or
    the default dtype of the category when every operand is a weak scalar)."""
    strong = [a for a in arrays_and_dtypes if not _is_weak_scalar(a)]
    weak_cat = max((_weak_category(a) for a in arrays_and_dtypes
                    if _is_weak_scalar(a)), default=-1)
    if strong:
        ret = onp.dtype(result_type(*strong)) if len(strong) > 1 \
            else _get_dtype(strong[0])
        ret = onp.dtype(ret)
        if weak_cat > _category_of_dtype(ret):
            return _default_dtype_for_category(weak_cat)
        return ret
    return _default_dtype_for_category(weak_cat)


def _to_numpy_result_type_arg(array_or_dtype):
    if isinstance(array_or_dtype, ndarray):
        return onp.empty((), dtype=array_or_dtype.dtype)
    return array_or_dtype


def result_type(*arrays_and_dtypes):
    """
    Returns the dtype that results from applying the type promotion rules to the arguments.
    Notes
    -----
    `result_type` is a standard API in
    https://data-apis.org/array-api/latest/API_specification/data_type_functions.html#result-type-arrays-and-dtypes
    instead of an official NumPy operator.
    Parameters
    ----------
    arrays_and_dtypes : mixed ndarrays and dtypes
        an arbitrary number of input arrays and/or dtypes.
    Returns
    -------
    out : dtype
        the dtype resulting from an operation involving the input arrays and dtypes.
    """
    if len(arrays_and_dtypes) > 0:
        if any(_is_weak_scalar(arg) for arg in arrays_and_dtypes):
            return _result_type_with_weak_scalars(arrays_and_dtypes)
        ret = _get_dtype(arrays_and_dtypes[0])
        for d in arrays_and_dtypes[1:]:
            dd = _get_dtype(d)
            if (ret, dd) in _type_promotion_table:
                ret = _type_promotion_table[ret, dd]
            elif (dd, ret) in _type_promotion_table:
                ret = _type_promotion_table[dd, ret]
            else:
                raise TypeError("Unknown type promotion between {} and {}".format(ret, dd))
        return ret
    raise ValueError("at least one array or dtype is required")
