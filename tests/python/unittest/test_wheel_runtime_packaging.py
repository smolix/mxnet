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


def _libinfo_version():
    libinfo = runpy.run_path(str(_repo_root() / "python" / "mxnet" / "libinfo.py"))
    return libinfo["__version__"]


def _setup_metadata(monkeypatch, tmp_path, cmake_cache, package_version=None):
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
    if package_version is None:
        monkeypatch.delenv("MXNET_PACKAGE_VERSION", raising=False)
    else:
        monkeypatch.setenv("MXNET_PACKAGE_VERSION", package_version)
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
    return captured


def _setup_install_requires(monkeypatch, tmp_path, cmake_cache):
    return _setup_metadata(monkeypatch, tmp_path, cmake_cache)["install_requires"]


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


def test_setup_metadata_uses_explicit_package_version(monkeypatch, tmp_path):
    metadata = _setup_metadata(
        monkeypatch,
        tmp_path,
        "USE_CUDA:BOOL=OFF\nUSE_OPENCV:BOOL=OFF\n",
        package_version="2.0.0+o11.override",
    )

    assert metadata["version"] == "2.0.0+o11.override"


def test_setup_metadata_normalizes_explicit_package_version(monkeypatch, tmp_path):
    metadata = _setup_metadata(
        monkeypatch,
        tmp_path,
        "USE_CUDA:BOOL=OFF\nUSE_OPENCV:BOOL=OFF\n",
        package_version="v2.0.0-rc1",
    )

    assert metadata["version"] == "2.0.0rc1"


def test_setup_metadata_rejects_invalid_package_version(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="MXNET_PACKAGE_VERSION.*PEP 440"):
        _setup_metadata(
            monkeypatch,
            tmp_path,
            "USE_CUDA:BOOL=OFF\nUSE_OPENCV:BOOL=OFF\n",
            package_version="refs/tags/v2.0.0",
        )


def test_setup_metadata_falls_back_to_libinfo_version(monkeypatch, tmp_path):
    metadata = _setup_metadata(
        monkeypatch,
        tmp_path,
        "USE_CUDA:BOOL=OFF\nUSE_OPENCV:BOOL=OFF\n",
    )

    assert metadata["version"] == _libinfo_version()


def test_setup_metadata_empty_package_version_falls_back_to_libinfo(monkeypatch, tmp_path):
    metadata = _setup_metadata(
        monkeypatch,
        tmp_path,
        "USE_CUDA:BOOL=OFF\nUSE_OPENCV:BOOL=OFF\n",
        package_version="  ",
    )

    assert metadata["version"] == _libinfo_version()


def test_release_wheel_build_enables_opencv_via_flavor():
    # The CI CPU wheel is built OpenCV-on through the shared build script's
    # linux-cpu flavor (USE_OPENCV + native bundling live there), not via an
    # inline cmake step — so CI and a local `MXNET_WHEEL_FLAVOR=cpu
    # tools/build_cleanup_wheel.sh` are the same recipe.
    workflow = (
        _repo_root() / ".github" / "workflows" / "release-wheel.yml"
    ).read_text()

    # OpenCV is now built in and its native libs bundled into the CPU wheel.
    assert "libopencv-dev" in workflow
    assert "MXNET_WHEEL_FLAVOR: cpu" in workflow
    assert "tools/build_cleanup_wheel.sh" in workflow
    # The old OpenCV-off opt-out must be gone.
    assert "-DUSE_OPENCV=OFF" not in workflow
    assert 'MXNET_SETUP_ENABLE_OPENCV_DEPS: "0"' not in workflow
    # Version resolution is preserved (the version is passed as the script's arg).
    assert "0.0.0.dev0+g${GITHUB_SHA::8}" in workflow
    assert "from packaging.version import Version" in workflow


def test_legacy_pip_setup_handles_opencv_runtime_policy():
    setup_py = (_repo_root() / "tools" / "pip" / "setup.py").read_text()

    assert "OPENCV_PYTHON_INSTALL_REQUIRES" in setup_py
    assert "MXNET_SETUP_ENABLE_OPENCV_DEPS" in setup_py
    assert "MXNET_SETUP_ALLOW_SYSTEM_OPENCV" in setup_py
    assert "_copy_opencv_libraries" in setup_py
    assert "libopencv_" in setup_py


def test_legacy_pip_setup_supports_package_version_override():
    setup_py = (_repo_root() / "tools" / "pip" / "setup.py").read_text()

    assert "MXNET_PACKAGE_VERSION" in setup_py
    assert "_package_version(libinfo['__version__'])" in setup_py
    assert "package_version_overridden" in setup_py


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
