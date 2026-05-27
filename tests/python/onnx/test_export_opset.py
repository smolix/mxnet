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

import numpy as np
import pytest
import mxnet as mx

onnx = pytest.importorskip("onnx")


def _default_domain_opset(model):
    versions = [
        opset.version
        for opset in model.opset_import
        if opset.domain in ("", "ai.onnx")
    ]
    assert len(versions) == 1
    return versions[0]


def _export_relu_file(tmp_path, **kwargs):
    data = mx.sym.Variable("data")
    sym = mx.sym.Activation(data=data, act_type="relu", name="relu")
    onnx_file = tmp_path / "relu.onnx"
    mx.onnx.export_model(
        sym,
        {},
        [(2, 3)],
        [np.float32],
        str(onnx_file),
        **kwargs
    )
    return onnx_file


def _export_relu(tmp_path, **kwargs):
    onnx_file = _export_relu_file(tmp_path, **kwargs)
    return onnx.load(str(onnx_file))


def test_export_model_defaults_to_mxnet_opset_13(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx.defs, "onnx_opset_version", lambda: 26)

    model = _export_relu(tmp_path)

    assert _default_domain_opset(model) == 13


def test_export_model_respects_explicit_opset_version(tmp_path):
    model = _export_relu(tmp_path, opset_version=12)

    assert _default_domain_opset(model) == 12


def test_default_opset_export_runs_in_onnxruntime(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    onnx_file = _export_relu_file(tmp_path)
    session = onnxruntime.InferenceSession(str(onnx_file))
    data = np.array([[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0]], dtype=np.float32)

    output = session.run(None, {session.get_inputs()[0].name: data})[0]

    np.testing.assert_array_equal(output, np.maximum(data, 0.0))
