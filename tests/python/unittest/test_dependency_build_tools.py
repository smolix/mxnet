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
import hashlib
import io
import inspect
import os
import subprocess
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


def _download_kwargs(module, expected_sha256):
    kwargs = {"expected_sha256": expected_sha256}
    if "retries" in inspect.signature(module.download).parameters:
        kwargs["retries"] = 1
    return kwargs


def _legacy_shell_download_function():
    script = (_repo_root() / "tools" / "dependencies" / "make_shared_dependencies.sh").read_text()
    start = script.index("download () {")
    end = script.index("\n\nif [[", start)
    return script[start:end]


def _fake_curl_function():
    return """curl () {
set -e
printf '%s\n' "$@" > "${CURL_ARGS_FILE}"
out_file=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        out_file="$2"
        shift 2
        continue
    fi
    shift
done
if [[ -n "${out_file}" ]]; then
    printf '%s' "${CURL_BODY}" > "${out_file}"
fi
return "${CURL_EXIT_CODE:-0}"
}
"""


def test_legacy_shell_download_quotes_url_and_output_path(tmp_path):
    out_file = tmp_path / "downloads with spaces" / "archive file.zip"
    args_file = tmp_path / "curl-args.txt"
    env = os.environ.copy()
    env.update({
        "CURL_ARGS_FILE": str(args_file),
        "CURL_BODY": "archive payload",
        "OUT_FILE": str(out_file),
    })
    script = (
        "set -e\n"
        f"{_fake_curl_function()}\n"
        f"{_legacy_shell_download_function()}\n"
        "mkdir -p \"$(dirname \"${OUT_FILE}\")\"\n"
        "download \"https://example.test/archive file.zip?token=a b\" \"${OUT_FILE}\"\n"
    )

    subprocess.run(["bash", "-c", script], env=env, check=True)

    assert out_file.read_text() == "archive payload"
    curl_args = args_file.read_text().splitlines()
    assert "--fail" in curl_args
    assert "https://example.test/archive file.zip?token=a b" in curl_args
    assert str(out_file) in curl_args


def test_legacy_shell_download_rejects_failed_curl_and_removes_partial(tmp_path):
    out_file = tmp_path / "archive.zip"
    args_file = tmp_path / "curl-args.txt"
    env = os.environ.copy()
    env.update({
        "CURL_ARGS_FILE": str(args_file),
        "CURL_BODY": "error page",
        "CURL_EXIT_CODE": "22",
        "OUT_FILE": str(out_file),
    })
    script = (
        "set -e\n"
        f"{_fake_curl_function()}\n"
        f"{_legacy_shell_download_function()}\n"
        "download \"https://example.test/missing.zip\" \"${OUT_FILE}\"\n"
    )

    result = subprocess.run(["bash", "-c", script], env=env, check=False)

    assert result.returncode == 1
    assert not out_file.exists()


@pytest.mark.parametrize(
    "script_name",
    ["build_openmp.py", "build_opencv.py", "build_libturbojpeg.py"],
)
def test_download_verifies_expected_sha256(script_name, monkeypatch, tmp_path):
    module = _load_tool(script_name)
    payload = b"known dependency archive"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    archive = tmp_path / "archive.bin"

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(payload),
    )

    module.download(
        "https://example.test/archive.bin",
        archive,
        **_download_kwargs(module, expected_sha256),
    )

    assert archive.read_bytes() == payload
    assert not archive.with_suffix(archive.suffix + ".tmp").exists()


@pytest.mark.parametrize(
    "script_name",
    ["build_openmp.py", "build_opencv.py", "build_libturbojpeg.py"],
)
def test_download_rejects_sha256_mismatch(script_name, monkeypatch, tmp_path):
    module = _load_tool(script_name)
    archive = tmp_path / "archive.bin"

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b"tampered archive"),
    )

    with pytest.raises(SystemExit, match="SHA256 mismatch"):
        module.download(
            "https://example.test/archive.bin",
            archive,
            **_download_kwargs(module, "0" * 64),
        )

    assert not archive.exists()
    assert not archive.with_suffix(archive.suffix + ".tmp").exists()


@pytest.mark.parametrize(
    "script_name",
    ["build_openmp.py", "build_opencv.py", "build_libturbojpeg.py"],
)
def test_download_cleans_partial_temp_file_on_stream_error(script_name, monkeypatch, tmp_path):
    module = _load_tool(script_name)
    archive = tmp_path / "archive.bin"

    class BrokenResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            raise OSError("stream interrupted")

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: BrokenResponse(),
    )
    monkeypatch.setattr(module.time, "sleep", lambda delay: None, raising=False)

    with pytest.raises(OSError, match="stream interrupted"):
        module.download(
            "https://example.test/archive.bin",
            archive,
            **_download_kwargs(module, "0" * 64),
        )

    assert not archive.exists()
    assert not archive.with_suffix(archive.suffix + ".tmp").exists()


def test_manifest_pins_default_dependency_urls():
    expected_urls = {
        "build_openmp.py": (
            "https://github.com/llvm/llvm-project/releases/download/"
            "llvmorg-22.1.5/llvm-project-22.1.5.src.tar.xz"
        ),
        "build_opencv.py": "https://github.com/opencv/opencv/archive/refs/tags/4.9.0.zip",
        "build_libturbojpeg.py": (
            "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.0.4.zip"
        ),
    }
    for script_name, url in expected_urls.items():
        module = _load_tool(script_name)
        checksum = module.expected_sha256_for_url(url)
        assert checksum is not None
        assert len(checksum) == 64


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
