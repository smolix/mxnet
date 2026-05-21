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
import runpy
import sys
import types
from pathlib import Path

import pytest


def _load_bundle_runtime_libs():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "python" / "tools" / "bundle_runtime_libs.py"
    spec = importlib.util.spec_from_file_location("bundle_runtime_libs", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _setup_install_requires(monkeypatch, tmp_path, cmake_cache):
    repo_root = _repo_root()
    python_dir = repo_root / "python"
    fake_build = tmp_path / "build"
    fake_build.mkdir()
    fake_libmxnet = fake_build / "libmxnet.so"
    fake_libmxnet.write_bytes(b"")
    (fake_build / "CMakeCache.txt").write_text(cmake_cache)

    captured = {}

    def fake_setup(**kwargs):
        captured.update(kwargs)

    class FakeDistribution:
        pass

    class FakeExtension:
        def __init__(self, *args, **kwargs):
            pass

    fake_setuptools = types.ModuleType("setuptools")
    fake_setuptools.find_packages = lambda exclude=None: []
    fake_setuptools.setup = fake_setup
    fake_setuptools.Distribution = FakeDistribution
    fake_extension = types.ModuleType("setuptools.extension")
    fake_extension.Extension = FakeExtension

    monkeypatch.chdir(python_dir)
    monkeypatch.setenv("MXNET_LIBRARY_PATH", str(fake_libmxnet))
    monkeypatch.setenv("MXNET_SETUP_SKIP_AUTOCOMPLETE", "1")
    monkeypatch.setenv("MXNET_SETUP_ENABLE_CUDA_DEPS", "0")
    monkeypatch.delenv("MXNET_SETUP_ENABLE_OPENCV_DEPS", raising=False)
    monkeypatch.setitem(sys.modules, "setuptools", fake_setuptools)
    monkeypatch.setitem(sys.modules, "setuptools.extension", fake_extension)
    monkeypatch.setattr(sys, "argv", ["setup.py", "egg_info"])

    original_path = list(sys.path)
    try:
        runpy.run_path(str(python_dir / "setup.py"), run_name="__main__")
    finally:
        sys.path[:] = original_path

    assert "install_requires" in captured
    return captured["install_requires"]


def test_setup_metadata_declares_opencv_python_dependency_when_enabled(monkeypatch, tmp_path):
    install_requires = _setup_install_requires(
        monkeypatch,
        tmp_path,
        "USE_CUDA:BOOL=OFF\nUSE_OPENCV:BOOL=ON\n",
    )

    assert "opencv-python>=4,<5" in install_requires


def test_setup_metadata_omits_opencv_python_dependency_when_disabled(monkeypatch, tmp_path):
    install_requires = _setup_install_requires(
        monkeypatch,
        tmp_path,
        "USE_CUDA:BOOL=OFF\nUSE_OPENCV:BOOL=OFF\n",
    )

    assert not any(req.startswith("opencv-python") for req in install_requires)


def test_release_wheel_build_disables_opencv_explicitly():
    workflow = (
        _repo_root() / ".github" / "workflows" / "release-wheel.yml"
    ).read_text()

    assert "libopencv-dev" not in workflow
    assert "-DUSE_OPENCV=OFF" in workflow
    assert "-DUSE_OPENCV=ON" not in workflow
    assert 'MXNET_SETUP_ENABLE_OPENCV_DEPS: "0"' in workflow
    assert 'cp -v "$libpath" python/mxnet/libmxnet.so' in workflow


def test_legacy_pip_setup_handles_opencv_runtime_policy():
    setup_py = (_repo_root() / "tools" / "pip" / "setup.py").read_text()

    assert "OPENCV_PYTHON_INSTALL_REQUIRES" in setup_py
    assert "MXNET_SETUP_ENABLE_OPENCV_DEPS" in setup_py
    assert "MXNET_SETUP_ALLOW_SYSTEM_OPENCV" in setup_py
    assert "_copy_opencv_libraries" in setup_py
    assert "libopencv_" in setup_py


def test_opencv_policy_rejects_silent_system_dependency():
    bundle_runtime_libs = _load_bundle_runtime_libs()

    with pytest.raises(RuntimeError, match="opencv-python.*does not satisfy"):
        bundle_runtime_libs.validate_opencv_policy(
            opencv_needed=["libopencv_imgcodecs.so.406"],
            bundled_opencv=[],
            drop_bundled=False,
            bundle_opencv=False,
            allow_system_opencv=False,
        )


def test_opencv_policy_allows_explicit_bundle_or_system_policy():
    bundle_runtime_libs = _load_bundle_runtime_libs()

    bundle_runtime_libs.validate_opencv_policy(
        opencv_needed=["libopencv_imgcodecs.so.406"],
        bundled_opencv=[],
        drop_bundled=True,
        bundle_opencv=True,
        allow_system_opencv=False,
    )
    bundle_runtime_libs.validate_opencv_policy(
        opencv_needed=["libopencv_imgcodecs.so.406"],
        bundled_opencv=[],
        drop_bundled=True,
        bundle_opencv=False,
        allow_system_opencv=True,
    )


def test_opencv_policy_preserves_existing_bundle_unless_dropped():
    bundle_runtime_libs = _load_bundle_runtime_libs()

    bundle_runtime_libs.validate_opencv_policy(
        opencv_needed=["libopencv_imgcodecs.so.406"],
        bundled_opencv=["libopencv_imgcodecs.so.406"],
        drop_bundled=False,
        bundle_opencv=False,
        allow_system_opencv=False,
    )
    with pytest.raises(RuntimeError):
        bundle_runtime_libs.validate_opencv_policy(
            opencv_needed=["libopencv_imgcodecs.so.406"],
            bundled_opencv=["libopencv_imgcodecs.so.406"],
            drop_bundled=True,
            bundle_opencv=False,
            allow_system_opencv=False,
        )


def test_opencv_policy_rejects_incomplete_existing_bundle():
    bundle_runtime_libs = _load_bundle_runtime_libs()

    with pytest.raises(RuntimeError):
        bundle_runtime_libs.validate_opencv_policy(
            opencv_needed=["libopencv_imgcodecs.so.406"],
            bundled_opencv=["libopencv_core.so.406"],
            drop_bundled=False,
            bundle_opencv=False,
            allow_system_opencv=False,
        )
