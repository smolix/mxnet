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


import sys as _sys
import ctypes as _ctypes
import numpy as np
from ..ndarray_doc import _build_doc
from libc.stdint cimport uint32_t, int64_t
from ..base import _LIB
from .. import _global_var

include "./base.pyi"

cdef class NDArrayBase:
    """Symbol is symbolic graph."""
    # handle for symbolic operator.
    cdef NDArrayHandle chandle
    cdef int cwritable
    cdef public bint _alive
    # Enable weak references, matching the ctypes NDArrayBase (a plain Python
    # class, which supports weakref implicitly). A Cython cdef class has no
    # weakref slot unless one is declared, so without this `weakref.ref(nd)`
    # raises "cannot create weak reference to 'NDArray' object" -- which breaks
    # the AMP cast weight cache (amp/amp.py) and any other weakref-based caching
    # whenever the fast cython path is compiled into the wheel.
    cdef object __weakref__

    cdef _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            if isinstance(handle, int):
                ptr = handle
            else:
                ptr = handle.value
            self.chandle = <SymbolHandle>(ptr)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return _ctypes.cast(<unsigned long long>self.chandle, _ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)
    property writable:
        def __get__(self):
            return bool(self.cwritable)

    def __init__(self, handle, writable=True):
        self._set_handle(handle)
        self.cwritable = writable
        self._alive = True

    def __del__(self):
        # tp_finalize: PEP 442 runs this during cyclic-GC collection, mirroring
        # the ctypes NDArrayBase.__del__. Without it, an NDArray caught in a
        # reference cycle only frees its backend handle at __dealloc__ (refcount
        # 0), which never happens while gc holds the cycle -- leaking the handle
        # and tripping the test suite's leak detector. Idempotent with
        # __dealloc__ via the NULL-after-free guard.
        if self.chandle != NULL:
            CALL(MXNDArrayFree(self.chandle))
            self.chandle = NULL
        self._alive = False

    def __dealloc__(self):
        # chandle may be NULL if __init__/__setstate__ raised before it was set
        # (e.g. the __reduce__/unpickle path constructs with None), or if __del__
        # already freed it during cycle collection; don't double-free NULL.
        if self.chandle != NULL:
            CALL(MXNDArrayFree(self.chandle))
            self.chandle = NULL
        self._alive = False

    def __reduce__(self):
        return (_global_var._ndarray_cls, (None,), self.__getstate__())

    def _get_handle(self):
        return <size_t>self.chandle


cdef NewArray(NDArrayHandle handle, int stype=-1, int is_np_array=0):
    """Create a new array given handle"""
    create_array_fn = _global_var._np_ndarray_cls if is_np_array else _global_var._ndarray_cls
    return create_array_fn(_ctypes.cast(<unsigned long long>handle, _ctypes.c_void_p), stype=stype)


def _imperative_invoke(handle, ndargs, keys, vals, out, is_np_op=0, output_is_list=0):
    """cython implementation of imperative invoke wrapper"""
    cdef unsigned long long ihandle = handle
    cdef OpHandle chandle = <OpHandle>ihandle
    cdef vector[string] ckeys
    cdef vector[string] cvals
    cdef vector[NDArrayHandle] ndvars
    cdef vector[NDArrayHandle] output_vars
    cdef NDArrayHandle* p_output_vars
    cdef NDArrayHandle ret_handle
    cdef int num_output
    cdef const int* p_output_stypes

    for i in ndargs:
        ndvars.push_back((<NDArrayBase>i).chandle)
    for i in keys:
        ckeys.push_back(c_str(i))
    for i in vals:
        cvals.push_back(c_str(str(i)))

    original_output = None
    if out is not None:
        original_output = out
        if isinstance(out, NDArrayBase):
            output_vars.push_back((<NDArrayBase>out).chandle)
        else:
            for i in out:
                output_vars.push_back((<NDArrayBase>i).chandle)

    num_output = output_vars.size()
    if output_vars.size() == 0:
        p_output_vars = NULL
    else:
        p_output_vars = &output_vars[0]

    cdef vector[const char*] param_keys = SVec2Ptr(ckeys)
    cdef vector[const char*] param_vals = SVec2Ptr(cvals)

    CALL(MXImperativeInvoke(
        chandle,
        <int>ndvars.size(),
        &ndvars[0] if ndvars.size() != 0 else NULL,
        &num_output,
        &p_output_vars,
        <int>param_keys.size(),
        CBeginPtr(param_keys),
        CBeginPtr(param_vals),
        &p_output_stypes))

    if original_output is not None:
        return original_output
    if num_output == 1 and not output_is_list:
        return NewArray(p_output_vars[0], p_output_stypes[0], is_np_op)
    else:
        return [NewArray(p_output_vars[i], p_output_stypes[i], is_np_op) for i in range(num_output)]
