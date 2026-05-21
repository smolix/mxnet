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
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_tool(monkeypatch):
    module_path = Path(__file__).resolve().parents[3] / "tools" / "im2rec.py"
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    spec = importlib.util.spec_from_file_location("im2rec", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_worker_closes_recordio(monkeypatch, tmp_path):
    module = _load_tool(monkeypatch)
    events = []

    class FakeRecord:
        def __init__(self, idx_path, rec_path, flag):
            events.append(("open", idx_path, rec_path, flag))

        def write_idx(self, idx, data):
            events.append(("write", idx, data))

        def close(self):
            events.append(("close",))

    class FakeQueue:
        def __init__(self):
            self.items = [
                (0, b"first", [10]),
                (1, None, [11]),
                (2, b"third", [12]),
                None,
            ]

        def get(self):
            return self.items.pop(0)

    monkeypatch.setattr(module.mx.recordio, "MXIndexedRecordIO", FakeRecord)

    module.write_worker(FakeQueue(), str(tmp_path / "train.lst"), str(tmp_path))

    assert events == [
        ("open", str(tmp_path / "train.idx"), str(tmp_path / "train.rec"), "w"),
        ("write", 10, b"first"),
        ("write", 12, b"third"),
        ("close",),
    ]
