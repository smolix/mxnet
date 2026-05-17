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
"""ONNX Export module"""

# Install onnx.mapping compatibility shim before any submodule looks it up.
# onnx >= 1.17 removed the ``onnx.mapping`` module; the exporter still
# references ``mapping.NP_TYPE_TO_TENSOR_TYPE`` / ``TENSOR_TYPE_TO_NP_TYPE``
# in many places.  Importing _onnx_compat patches ``onnx.mapping`` back in
# when missing, regardless of which onnx version is installed.
try:
    from . import _onnx_compat  # noqa: F401
except ImportError:
    # onnx itself is optional; the exporter will raise a clear error later
    # if onnx isn't installed.
    pass

from ._export_model import export_model, get_operator_support
from ._op_translations import _op_translations_opset12
from ._op_translations import _op_translations_opset13
