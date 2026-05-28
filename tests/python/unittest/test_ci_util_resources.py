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


def _load_ci_util():
    module_path = Path(__file__).resolve().parents[3] / "ci" / "util.py"
    spec = importlib.util.spec_from_file_location("ci_util", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_file_closes_streamed_response(monkeypatch, tmp_path):
    module = _load_ci_util()

    class FakeResponse:
        status_code = 200

        def __init__(self):
            self.closed = False

        def iter_content(self, chunk_size):
            assert chunk_size == 1024
            yield b"abc"

        def close(self):
            self.closed = True

    response = FakeResponse()

    def fake_get(url, **kwargs):
        assert kwargs["stream"] is True
        assert kwargs["timeout"] == module.DOWNLOAD_TIMEOUT_SECONDS
        return response

    monkeypatch.setattr(module.requests, "get", fake_get)

    path = module.download_file("https://example.com/file.bin", str(tmp_path))

    assert Path(path).read_bytes() == b"abc"
    assert response.closed
