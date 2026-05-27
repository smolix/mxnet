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

# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments
# pylint: disable=global-statement, unused-import
"""NDArray configuration API."""

import ctypes

from ..base import _LIB
from ..base import c_str_array, c_handle_array
from ..base import NDArrayHandle
from ..base import check_call
from .. import _global_var

class NDArrayBase(object):
    """Base data structure for ndarray"""
    __slots__ = ["handle", "writable", "_alive", "__weakref__"]
    # pylint: disable= no-member

    def __init__(self, handle, writable=True):
        """initialize a new NDArray

        Parameters
        ----------
        handle : NDArrayHandle
            NDArray handle of C API
        """
        if handle is not None:
            assert isinstance(handle, NDArrayHandle)
        self.handle = handle
        self.writable = writable
        self._alive = True

    def __del__(self):
        check_call(_LIB.MXNDArrayFree(self.handle))
        self._alive = False

    def __reduce__(self):
        return (_global_var._ndarray_cls, (None,), self.__getstate__())


def _make_ndarray_outputs(output_vars, out_stypes, num_output, create_ndarray_fn, output_is_list):
    """Create Python NDArrays and free unwrapped handles if wrapping raises."""
    wrapped_count = 0
    try:
        ret = []
        for i in range(num_output):
            ret.append(create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle),
                                         stype=out_stypes[i]))
            wrapped_count += 1
        if num_output == 1 and not output_is_list:
            return ret[0]
        return ret
    except Exception:
        for i in range(wrapped_count, num_output):
            check_call(_LIB.MXNDArrayFree(output_vars[i]))
        raise


def _imperative_invoke(handle, ndargs, keys, vals, out, is_np_op, output_is_list):
    """ctypes implementation of imperative invoke wrapper"""
    if out is not None:
        original_output = out
        if isinstance(out, NDArrayBase):
            out = (out,)
        num_output = ctypes.c_int(len(out))
        output_vars = c_handle_array(out)
        output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
    else:
        original_output = None
        output_vars = ctypes.POINTER(NDArrayHandle)()
        num_output = ctypes.c_int(0)

    # return output stypes to avoid the c_api call for checking
    # a handle's stype in _ndarray_cls
    out_stypes = ctypes.POINTER(ctypes.c_int)()

    check_call(_LIB.MXImperativeInvoke(
        ctypes.c_void_p(handle),
        ctypes.c_int(len(ndargs)),
        c_handle_array(ndargs),
        ctypes.byref(num_output),
        ctypes.byref(output_vars),
        ctypes.c_int(len(keys)),
        c_str_array(keys),
        c_str_array([str(s) for s in vals]),
        ctypes.byref(out_stypes)))

    create_ndarray_fn = _global_var._np_ndarray_cls if is_np_op else _global_var._ndarray_cls
    if original_output is not None:
        return original_output
    return _make_ndarray_outputs(output_vars, out_stypes, num_output.value,
                                 create_ndarray_fn, output_is_list)
