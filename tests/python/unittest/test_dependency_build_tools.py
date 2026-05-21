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
import io
import tarfile
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


def _write_tar_file(tf, name, data):
    payload = data.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


@pytest.fixture
def force_legacy_tar_extractall(monkeypatch):
    original_extractall = tarfile.TarFile.extractall

    def extractall_without_filter(self, path=".", members=None, *, numeric_owner=False, filter=None):
        if filter is not None:
            raise TypeError("extractall() got an unexpected keyword argument 'filter'")
        return original_extractall(
            self, path, members=members, numeric_owner=numeric_owner, filter="fully_trusted"
        )

    monkeypatch.setattr(tarfile.TarFile, "extractall", extractall_without_filter)


def test_openmp_tar_extract_legacy_fallback_allows_archive_contents_under_destination(
    tmp_path, force_legacy_tar_extractall
):
    module = _load_tool("build_openmp.py")
    archive = tmp_path / "archive.tar"
    source_dir = tmp_path / "src" / "llvm-project-1.0.src"

    with tarfile.open(archive, "w") as tf:
        _write_tar_file(tf, "llvm-project-1.0.src/README.md", "ok")

    module.extract(archive, source_dir)

    assert (source_dir / "README.md").read_text() == "ok"


def test_openmp_tar_extract_legacy_fallback_rejects_path_traversal(
    tmp_path, force_legacy_tar_extractall
):
    module = _load_tool("build_openmp.py")
    archive = tmp_path / "archive.tar"
    source_dir = tmp_path / "src" / "llvm-project-1.0.src"
    outside_file = tmp_path / "pwned.txt"

    with tarfile.open(archive, "w") as tf:
        _write_tar_file(tf, "llvm-project-1.0.src/README.md", "ok")
        _write_tar_file(tf, "../pwned.txt", "bad")

    with pytest.raises(SystemExit, match="Unsafe archive member path"):
        module.extract(archive, source_dir)

    assert not outside_file.exists()
    assert not source_dir.exists()
