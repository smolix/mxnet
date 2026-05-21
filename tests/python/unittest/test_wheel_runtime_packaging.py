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


def _load_bundle_runtime_libs():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "python" / "tools" / "bundle_runtime_libs.py"
    spec = importlib.util.spec_from_file_location("bundle_runtime_libs", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_opencv_policy_rejects_silent_system_dependency():
    bundle_runtime_libs = _load_bundle_runtime_libs()

    with pytest.raises(RuntimeError, match="install_requires cannot express"):
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
