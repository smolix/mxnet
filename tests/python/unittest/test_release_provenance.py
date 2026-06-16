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


def _load_release_provenance():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools" / "release_provenance.py"
    spec = importlib.util.spec_from_file_location("release_provenance", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_repo(tmp_path, version):
    libinfo = tmp_path / "python" / "mxnet" / "libinfo.py"
    libinfo.parent.mkdir(parents=True)
    libinfo.write_text('__version__ = "{}"\n'.format(version))
    cache = tmp_path / "build" / "CMakeCache.txt"
    cache.parent.mkdir()
    cache.write_text(
        "USE_CUDA:BOOL=ON\n"
        "USE_CUDNN:BOOL=ON\n"
        "USE_NCCL:BOOL=ON\n"
        "USE_ONEDNN:BOOL=ON\n"
        "USE_OPENCV:BOOL=OFF\n"
        "USE_OPENMP:BOOL=ON\n"
    )
    return cache


def test_collect_provenance_reports_release_staging_fields(monkeypatch, tmp_path):
    release_provenance = _load_release_provenance()
    version = "2.0.0+cu13.bw.20260518.1"
    _write_minimal_repo(tmp_path, version)
    wheel = tmp_path / "dist" / (
        "mxnet-{}-cp312-cp312-linux_x86_64.whl".format(version)
    )
    wheel.parent.mkdir()
    wheel.write_bytes(b"")

    def fake_git_output(repo_root, args):
        assert repo_root == tmp_path
        if args == ["rev-parse", "HEAD"]:
            return "0123456789abcdef0123456789abcdef01234567"
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            return ""
        if args == ["status", "--porcelain", "--untracked-files=normal"]:
            return "?? dist/"
        raise AssertionError("unexpected git args: {}".format(args))

    monkeypatch.setattr(release_provenance, "_git_output", fake_git_output)
    monkeypatch.setattr(
        release_provenance,
        "inspect_wheel_payload",
        lambda wheel_path: {
            "inspected": True,
            "error": None,
            "has_libmxnet": True,
            "needed": [],
            "cudnn_needed": [],
            "nccl_needed": [],
            "opencv_needed": [],
            "opencv_bundled": [],
            "opencv_bundled_sonames": [],
            "runpath": None,
            "runpath_has_origin_lib": False,
            "runpath_has_nvidia_cudnn": False,
            "runpath_has_nvidia_nccl": False,
        },
    )

    report = release_provenance.collect_provenance(
        repo_root=tmp_path,
        wheel_paths=[wheel],
    )

    assert report["git"]["commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert report["git"]["dirty"] is False
    assert report["git"]["untracked_count"] == 1
    assert report["package"] == {
        "version": version,
        "source": "python/mxnet/libinfo.py",
    }
    assert report["features"]["USE_CUDA"] == {"enabled": True, "raw": "ON"}
    assert report["features"]["USE_OPENCV"] == {"enabled": False, "raw": "OFF"}
    assert report["wheels"][0]["version_matches_package"] is True
    assert report["wheels"][0]["distribution_matches_package"] is True
    assert release_provenance.validate_provenance(report) == []


def test_validate_provenance_rejects_dirty_tree_and_wheel_version_mismatch():
    release_provenance = _load_release_provenance()
    report = {
        "expected_package_name": "mxnet",
        "git": {
            "commit": "fedcba9876543210fedcba9876543210fedcba98",
            "short_commit": "fedcba987654",
            "dirty": True,
            "tracked_change_count": 2,
            "untracked_count": 0,
        },
        "package": {"version": "2.0.0"},
        "features": {
            "cache_found": True,
            "cache_path": "build/CMakeCache.txt",
            "USE_CUDA": {"enabled": True, "raw": "ON"},
            "USE_CUDNN": {"enabled": True, "raw": "ON"},
            "USE_NCCL": {"enabled": True, "raw": "ON"},
            "USE_ONEDNN": {"enabled": True, "raw": "ON"},
            "USE_OPENCV": {"enabled": False, "raw": "OFF"},
            "USE_OPENMP": {"enabled": True, "raw": "ON"},
        },
        "wheels": [
            {
                "filename": "mxnet-2.0.1-cp312-cp312-linux_x86_64.whl",
                "path": "dist/mxnet-2.0.1-cp312-cp312-linux_x86_64.whl",
                "exists": True,
                "distribution": "mxnet",
                "distribution_matches_package": True,
                "version": "2.0.1",
                "version_matches_package": False,
            }
        ],
    }

    errors = release_provenance.validate_provenance(report)

    assert any("tracked working tree is dirty" in error for error in errors)
    assert any("version 2.0.1 does not match package version 2.0.0" in error
               for error in errors)


def test_validate_provenance_checks_expected_feature_flags(monkeypatch, tmp_path):
    release_provenance = _load_release_provenance()
    version = "2.0.0"
    _write_minimal_repo(tmp_path, version)

    def fake_git_output(repo_root, args):
        if args == ["rev-parse", "HEAD"]:
            return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        return ""

    monkeypatch.setattr(release_provenance, "_git_output", fake_git_output)
    report = release_provenance.collect_provenance(
        repo_root=tmp_path,
        wheel_paths=[],
    )

    errors = release_provenance.validate_provenance(
        report,
        expect_cuda="off",
        expect_opencv="off",
    )

    assert any("USE_CUDA expected OFF, found ON" in error for error in errors)
    assert not any("USE_OPENCV expected OFF" in error for error in errors)


def test_validate_provenance_rejects_stale_cmake_commit_stamp():
    release_provenance = _load_release_provenance()
    report = {
        "expected_package_name": "mxnet",
        "git": {
            "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "short_commit": "aaaaaaaaaaaa",
            "dirty": False,
            "tracked_change_count": 0,
            "untracked_count": 0,
        },
        "package": {"version": "2.0.0"},
        "features": {
            "cache_found": True,
            "cache_path": "build/CMakeCache.txt",
            "USE_CUDA": {"enabled": True, "raw": "ON"},
            "USE_CUDNN": {"enabled": True, "raw": "ON"},
            "USE_NCCL": {"enabled": True, "raw": "ON"},
            "USE_ONEDNN": {"enabled": True, "raw": "ON"},
            "USE_OPENCV": {"enabled": False, "raw": "OFF"},
            "USE_OPENMP": {"enabled": True, "raw": "ON"},
        },
        "build": {
            "build_dir": "build",
            "metadata_found": True,
            "commit_hashes": ["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
            "sources": [],
        },
        "wheels": [],
    }

    errors = release_provenance.validate_provenance(report)

    assert any("does not match git HEAD" in error for error in errors)


def test_collect_provenance_reads_cmake_commit_stamp(monkeypatch, tmp_path):
    release_provenance = _load_release_provenance()
    version = "2.0.0"
    _write_minimal_repo(tmp_path, version)
    build_ninja = tmp_path / "build" / "build.ninja"
    build_ninja.write_text(
        'DEFINES = -DMXNET_COMMIT_HASH=\\"'
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        '\\" -DMXNET_BRANCH=\\"main\\"\n'
    )

    def fake_git_output(repo_root, args):
        if args == ["rev-parse", "HEAD"]:
            return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        return ""

    monkeypatch.setattr(release_provenance, "_git_output", fake_git_output)
    report = release_provenance.collect_provenance(repo_root=tmp_path)

    assert report["build"]["metadata_found"] is True
    assert report["build"]["commit_hashes"] == [
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ]
    assert release_provenance.validate_provenance(report) == []


def test_validate_provenance_checks_opencv_wheel_payload():
    release_provenance = _load_release_provenance()
    report = {
        "expected_package_name": "mxnet",
        "git": {
            "commit": "fedcba9876543210fedcba9876543210fedcba98",
            "short_commit": "fedcba987654",
            "dirty": False,
            "tracked_change_count": 0,
            "untracked_count": 0,
        },
        "package": {"version": "2.0.0"},
        "features": {
            "cache_found": True,
            "cache_path": "build/CMakeCache.txt",
            "USE_CUDA": {"enabled": True, "raw": "ON"},
            "USE_CUDNN": {"enabled": True, "raw": "ON"},
            "USE_NCCL": {"enabled": True, "raw": "ON"},
            "USE_ONEDNN": {"enabled": True, "raw": "ON"},
            "USE_OPENCV": {"enabled": True, "raw": "ON"},
            "USE_OPENMP": {"enabled": True, "raw": "ON"},
        },
        "wheels": [
            {
                "filename": "mxnet-2.0.0-cp312-cp312-linux_x86_64.whl",
                "path": "dist/mxnet-2.0.0-cp312-cp312-linux_x86_64.whl",
                "exists": True,
                "distribution": "mxnet",
                "distribution_matches_package": True,
                "version": "2.0.0",
                "version_matches_package": True,
                "payload": {
                    "inspected": True,
                    "error": None,
                    "has_libmxnet": True,
                    "needed": ["libcudart.so.13"],
                    "cudnn_needed": [],
                    "nccl_needed": [],
                    "opencv_needed": [],
                    "opencv_bundled": [],
                    "opencv_bundled_sonames": [],
                    "runpath": "$ORIGIN/../nvidia/cu13/lib",
                    "runpath_has_origin_lib": False,
                    "runpath_has_nvidia_cudnn": False,
                    "runpath_has_nvidia_nccl": False,
                },
            }
        ],
    }

    errors = release_provenance.validate_provenance(
        report,
        expect_cuda="on",
        expect_opencv="on",
    )

    assert any("has no libopencv_* NEEDED entries" in error for error in errors)
    assert any("does not bundle mxnet/lib/libopencv_*" in error for error in errors)
    assert any("RUNPATH does not include $ORIGIN/lib" in error for error in errors)


def test_validate_provenance_accepts_opencv_wheel_payload():
    release_provenance = _load_release_provenance()
    report = {
        "expected_package_name": "mxnet",
        "git": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "short_commit": "0123456789ab",
            "dirty": False,
            "tracked_change_count": 0,
            "untracked_count": 0,
        },
        "package": {"version": "2.0.0"},
        "features": {
            "cache_found": True,
            "cache_path": "build/CMakeCache.txt",
            "USE_CUDA": {"enabled": True, "raw": "ON"},
            "USE_CUDNN": {"enabled": True, "raw": "ON"},
            "USE_NCCL": {"enabled": True, "raw": "ON"},
            "USE_ONEDNN": {"enabled": True, "raw": "ON"},
            "USE_OPENCV": {"enabled": True, "raw": "ON"},
            "USE_OPENMP": {"enabled": True, "raw": "ON"},
        },
        "wheels": [
            {
                "filename": "mxnet-2.0.0-cp312-cp312-linux_x86_64.whl",
                "path": "dist/mxnet-2.0.0-cp312-cp312-linux_x86_64.whl",
                "exists": True,
                "distribution": "mxnet",
                "distribution_matches_package": True,
                "version": "2.0.0",
                "version_matches_package": True,
                "payload": {
                    "inspected": True,
                    "error": None,
                    "has_libmxnet": True,
                    "needed": [
                        "libcudnn.so.9",
                        "libnccl.so.2",
                        "libopencv_imgcodecs.so.406",
                        "libopencv_imgproc.so.406",
                    ],
                    "cudnn_needed": ["libcudnn.so.9"],
                    "nccl_needed": ["libnccl.so.2"],
                    "opencv_needed": [
                        "libopencv_imgcodecs.so.406",
                        "libopencv_imgproc.so.406",
                    ],
                    "opencv_bundled": [
                        "mxnet/lib/libopencv_imgcodecs.so.406",
                        "mxnet/lib/libopencv_imgproc.so.406",
                    ],
                    "opencv_bundled_sonames": [
                        "libopencv_imgcodecs.so.406",
                        "libopencv_imgproc.so.406",
                    ],
                    "runpath": (
                        "$ORIGIN/lib:$ORIGIN/../nvidia/cudnn/lib:"
                        "$ORIGIN/../nvidia/nccl/lib:$ORIGIN/../nvidia/cu13/lib"
                    ),
                    "runpath_has_origin_lib": True,
                    "runpath_has_nvidia_cudnn": True,
                    "runpath_has_nvidia_nccl": True,
                },
            }
        ],
    }

    assert release_provenance.validate_provenance(
        report,
        expect_cuda="on",
        expect_cudnn="on",
        expect_nccl="on",
        expect_onednn="on",
        expect_opencv="on",
    ) == []


def test_validate_provenance_checks_cudnn_and_nccl_wheel_payload():
    release_provenance = _load_release_provenance()
    report = {
        "expected_package_name": "mxnet",
        "git": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "short_commit": "0123456789ab",
            "dirty": False,
            "tracked_change_count": 0,
            "untracked_count": 0,
        },
        "package": {"version": "2.0.0"},
        "features": {
            "cache_found": True,
            "cache_path": "build/CMakeCache.txt",
            "USE_CUDA": {"enabled": True, "raw": "ON"},
            "USE_CUDNN": {"enabled": True, "raw": "ON"},
            "USE_NCCL": {"enabled": True, "raw": "ON"},
            "USE_ONEDNN": {"enabled": True, "raw": "ON"},
            "USE_OPENCV": {"enabled": False, "raw": "OFF"},
            "USE_OPENMP": {"enabled": True, "raw": "ON"},
        },
        "wheels": [
            {
                "filename": "mxnet-2.0.0-cp312-cp312-linux_x86_64.whl",
                "path": "dist/mxnet-2.0.0-cp312-cp312-linux_x86_64.whl",
                "exists": True,
                "distribution": "mxnet",
                "distribution_matches_package": True,
                "version": "2.0.0",
                "version_matches_package": True,
                "payload": {
                    "inspected": True,
                    "error": None,
                    "has_libmxnet": True,
                    "needed": ["libcudnn.so.9"],
                    "cudnn_needed": ["libcudnn.so.9"],
                    "nccl_needed": [],
                    "opencv_needed": [],
                    "opencv_bundled": [],
                    "opencv_bundled_sonames": [],
                    "runpath": "$ORIGIN/../nvidia/cu13/lib",
                    "runpath_has_origin_lib": False,
                    "runpath_has_nvidia_cudnn": False,
                    "runpath_has_nvidia_nccl": False,
                },
            }
        ],
    }

    errors = release_provenance.validate_provenance(
        report,
        expect_cuda="on",
        expect_cudnn="on",
        expect_nccl="on",
        expect_onednn="on",
        expect_opencv="off",
    )

    assert any("RUNPATH does not include $ORIGIN/../nvidia/cudnn/lib" in error
               for error in errors)
    assert any("has no libnccl NEEDED entries" in error for error in errors)
    assert any("RUNPATH does not include $ORIGIN/../nvidia/nccl/lib" in error
               for error in errors)


def _macos_wheel_report(openmp_bundled, openmp_needed, openmp_loader_relative):
    """A macOS arm64 CPU-wheel provenance report (oneDNN+OpenCV+OpenMP, onnx hard)."""
    return {
        "expected_package_name": "mxnet",
        "git": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "short_commit": "0123456789ab",
            "dirty": False,
            "tracked_change_count": 0,
            "untracked_count": 0,
        },
        "package": {"version": "2.0.0"},
        "features": {
            "cache_found": True,
            "cache_path": "build/CMakeCache.txt",
            "USE_CUDA": {"enabled": False, "raw": "OFF"},
            "USE_CUDNN": {"enabled": False, "raw": "OFF"},
            "USE_NCCL": {"enabled": False, "raw": "OFF"},
            "USE_ONEDNN": {"enabled": True, "raw": "ON"},
            "USE_OPENCV": {"enabled": True, "raw": "ON"},
            "USE_OPENMP": {"enabled": True, "raw": "ON"},
        },
        "wheels": [
            {
                "filename": "mxnet-2.0.0-cp312-cp312-macosx_26_0_arm64.whl",
                "path": "dist/mxnet-2.0.0-cp312-cp312-macosx_26_0_arm64.whl",
                "exists": True,
                "distribution": "mxnet",
                "distribution_matches_package": True,
                "version": "2.0.0",
                "version_matches_package": True,
                "payload": {
                    "inspected": True,
                    "error": None,
                    "format": "macho",
                    "has_libmxnet": True,
                    "needed": (
                        ["libomp.dylib"] if openmp_needed else []
                    ) + ["libopencv_imgcodecs.dylib"],
                    "cudnn_needed": [],
                    "nccl_needed": [],
                    "opencv_needed": ["libopencv_imgcodecs.dylib"],
                    "opencv_bundled": ["mxnet/lib/libopencv_imgcodecs.dylib"],
                    "opencv_bundled_sonames": ["libopencv_imgcodecs.dylib"],
                    "openmp_needed": ["libomp.dylib"] if openmp_needed else [],
                    "openmp_bundled": (
                        ["mxnet/lib/libomp.dylib"] if openmp_bundled else []
                    ),
                    "openmp_runtime_loader_relative": openmp_loader_relative,
                    "onnx_pkg_files": ["mxnet/onnx/__init__.py"],
                    "onnx_hard_require": True,
                    "onnx_extra_require": True,
                    "metadata_found": True,
                    "runpath": "@loader_path/lib",
                    "runpath_has_origin_lib": True,
                    "runpath_has_nvidia_cudnn": False,
                    "runpath_has_nvidia_nccl": False,
                },
            }
        ],
    }


def test_validate_provenance_accepts_macos_openmp_wheel_payload():
    release_provenance = _load_release_provenance()
    report = _macos_wheel_report(
        openmp_bundled=True, openmp_needed=True, openmp_loader_relative=True)
    assert release_provenance.validate_provenance(
        report,
        expect_cuda="off",
        expect_cudnn="off",
        expect_nccl="off",
        expect_onednn="on",
        expect_opencv="on",
        expect_onnx="on",
        expect_openmp="on",
    ) == []


def test_validate_provenance_rejects_macos_wheel_missing_bundled_openmp():
    release_provenance = _load_release_provenance()
    report = _macos_wheel_report(
        openmp_bundled=False, openmp_needed=False, openmp_loader_relative=False)
    errors = release_provenance.validate_provenance(
        report,
        expect_cuda="off",
        expect_cudnn="off",
        expect_nccl="off",
        expect_onednn="on",
        expect_opencv="on",
        expect_onnx="on",
        expect_openmp="on",
    )
    assert any("has no OpenMP runtime dependency" in error for error in errors)
    assert any("does not bundle the OpenMP runtime" in error for error in errors)
    assert any("not loader-relative" in error for error in errors)


def test_validate_provenance_rejects_openmp_build_flag_off():
    release_provenance = _load_release_provenance()
    report = _macos_wheel_report(
        openmp_bundled=True, openmp_needed=True, openmp_loader_relative=True)
    report["features"]["USE_OPENMP"] = {"enabled": False, "raw": "OFF"}
    errors = release_provenance.validate_provenance(
        report, expect_openmp="on")
    assert any("USE_OPENMP expected ON, found OFF" in error for error in errors)
