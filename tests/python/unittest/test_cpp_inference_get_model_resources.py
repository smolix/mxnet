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

import importlib.util
from pathlib import Path

import pytest


def _load_get_model():
    module_path = (Path(__file__).resolve().parents[3] / "cpp-package" / "example" /
                   "inference" / "multi_threaded_inference" / "get_model.py")
    spec = importlib.util.spec_from_file_location("cpp_inference_get_model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_removes_partial_file_and_closes_response(monkeypatch, tmp_path):
    module = _load_get_model()

    class FakeResponse:
        status_code = 200

        def __init__(self):
            self.closed = False

        def iter_content(self, chunk_size):
            yield b"partial"
            raise RuntimeError("stream failed")

        def close(self):
            self.closed = True

    response = FakeResponse()

    def fake_get(url, **kwargs):
        assert kwargs["stream"] is True
        assert kwargs["timeout"] == 30
        return response

    monkeypatch.setattr(module.requests, "get", fake_get)

    target = tmp_path / "model.params"
    with pytest.raises(RuntimeError, match="stream failed"):
        module.download("https://example.com/model.params", fname=str(target), retries=1)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
    assert response.closed
