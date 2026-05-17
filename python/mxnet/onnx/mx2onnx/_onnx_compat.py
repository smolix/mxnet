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

"""Compatibility shim for onnx >= 1.17.

The ``onnx.mapping`` module was removed in onnx 1.17 in favour of helper
functions on :mod:`onnx.helper`.  MXNet's ONNX exporter still references
``onnx.mapping.NP_TYPE_TO_TENSOR_TYPE`` and ``onnx.mapping.TENSOR_TYPE_TO_NP_TYPE``
in many places.  Importing this module ensures ``onnx.mapping`` is available with
the two dictionaries the exporter relies on, regardless of which onnx version is
installed.
"""

import sys
import types

import numpy as _np
import onnx as _onnx


def _build_mapping_module():
    """Construct an ``onnx.mapping``-compatible module from new helper APIs."""
    helper = _onnx.helper
    mod = types.ModuleType('onnx.mapping')

    np_to_tensor = {}
    tensor_to_np = {}

    # Prefer the modern helper APIs (onnx >= 1.13).  Fall back gracefully if
    # individual dtypes are unsupported by the installed onnx build.
    get_all = getattr(helper, 'get_all_tensor_dtypes', None)
    t2np = getattr(helper, 'tensor_dtype_to_np_dtype', None)
    np2t = getattr(helper, 'np_dtype_to_tensor_dtype', None)

    if get_all is not None and t2np is not None and np2t is not None:
        for tensor_dtype in get_all():
            try:
                np_dtype = t2np(tensor_dtype)
            except Exception:  # pylint: disable=broad-except
                continue
            tensor_to_np[tensor_dtype] = np_dtype
            try:
                np_to_tensor[_np.dtype(np_dtype)] = tensor_dtype
            except Exception:  # pylint: disable=broad-except
                pass

    mod.NP_TYPE_TO_TENSOR_TYPE = np_to_tensor
    mod.TENSOR_TYPE_TO_NP_TYPE = tensor_to_np
    return mod


# Only install the shim if onnx.mapping is missing (older onnx still ships it).
if not hasattr(_onnx, 'mapping'):
    _shim = _build_mapping_module()
    _onnx.mapping = _shim
    sys.modules['onnx.mapping'] = _shim
