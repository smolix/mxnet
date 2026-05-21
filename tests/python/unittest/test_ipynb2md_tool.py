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
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_tool(monkeypatch):
    module_path = Path(__file__).resolve().parents[3] / "tools" / "ipynb2md.py"
    monkeypatch.setitem(sys.modules, "nbformat", SimpleNamespace(NO_CONVERT=object()))
    spec = importlib.util.spec_from_file_location("ipynb2md", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_notebook_uses_argument_list_and_cleans_temp_file(monkeypatch, tmp_path):
    module = _load_tool(monkeypatch)
    old_ipynb = tmp_path / "input;touch pwned.ipynb"
    md_file = tmp_path / "out;touch pwned.md"
    temp_ipynb = tmp_path / "tmp;touch pwned.ipynb"
    calls = []
    unlinked = []

    class FakeTempFile:
        name = str(temp_ipynb)

        def close(self):
            pass

    monkeypatch.setattr(
        module.tempfile, "NamedTemporaryFile", lambda **kwargs: FakeTempFile())
    monkeypatch.setattr(
        module,
        "clear_notebook",
        lambda old_path, new_path: calls.append(("clear", old_path, new_path)))
    monkeypatch.setattr(
        module.subprocess, "run", lambda argv, check: calls.append(("run", argv, check)))
    monkeypatch.setattr(module.os, "unlink", lambda path: unlinked.append(path))

    module.convert_notebook(str(old_ipynb), str(md_file))

    assert calls == [
        ("clear", str(old_ipynb), str(temp_ipynb)),
        ("run",
         ["jupyter", "nbconvert", str(temp_ipynb), "--to", "markdown", "--output", str(md_file)],
         True),
    ]
    assert unlinked == [str(temp_ipynb)]
    assert md_file.read_text() == "<!-- INSERT SOURCE DOWNLOAD BUTTONS -->"


def test_convert_notebook_cleans_temp_file_when_nbconvert_fails(monkeypatch, tmp_path):
    module = _load_tool(monkeypatch)
    temp_ipynb = tmp_path / "tmp.ipynb"
    unlinked = []

    class FakeTempFile:
        name = str(temp_ipynb)

        def close(self):
            pass

    def fail_nbconvert(argv, check):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(
        module.tempfile, "NamedTemporaryFile", lambda **kwargs: FakeTempFile())
    monkeypatch.setattr(module, "clear_notebook", lambda old_path, new_path: None)
    monkeypatch.setattr(module.subprocess, "run", fail_nbconvert)
    monkeypatch.setattr(module.os, "unlink", lambda path: unlinked.append(path))

    with pytest.raises(subprocess.CalledProcessError):
        module.convert_notebook(str(tmp_path / "input.ipynb"), str(tmp_path / "out.md"))

    assert unlinked == [str(temp_ipynb)]
