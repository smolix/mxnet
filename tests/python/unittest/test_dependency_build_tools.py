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
import zipfile
from pathlib import Path

import pytest


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _load_tool(script_name):
    module_path = _repo_root() / "tools" / "dependencies" / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script_name", ["build_opencv.py", "build_libturbojpeg.py"])
def test_zip_extract_allows_archive_contents_under_destination(script_name, tmp_path):
    module = _load_tool(script_name)
    archive = tmp_path / "archive.zip"
    source_dir = tmp_path / "src" / "project-1.0"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("project-1.0/README.md", "ok")

    module.extract(archive, source_dir)

    assert (source_dir / "README.md").read_text() == "ok"


@pytest.mark.parametrize("script_name", ["build_opencv.py", "build_libturbojpeg.py"])
def test_zip_extract_rejects_path_traversal(script_name, tmp_path):
    module = _load_tool(script_name)
    archive = tmp_path / "archive.zip"
    source_dir = tmp_path / "src" / "project-1.0"
    outside_file = tmp_path / "pwned.txt"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("project-1.0/README.md", "ok")
        zf.writestr("../pwned.txt", "bad")

    with pytest.raises(SystemExit, match="Unsafe archive member path"):
        module.extract(archive, source_dir)

    assert not outside_file.exists()
    assert not source_dir.exists()
