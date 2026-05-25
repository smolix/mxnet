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
"""Runtime repairs for CUDA Python wheel layouts."""

from __future__ import absolute_import

import os
import sys
from pathlib import Path

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    import importlib_metadata  # type: ignore


_CUDNN_DISTRIBUTIONS = (
    "nvidia-cudnn-cu13",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-cu11",
)


def _cudnn_package_version():
    for dist_name in _CUDNN_DISTRIBUTIONS:
        try:
            return importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _cudnn_alias_suffix(version):
    parts = version.split(".")
    if len(parts) < 3:
        return None
    major, minor, patch = parts[:3]
    if not (major.isdigit() and minor.isdigit() and patch.isdigit()):
        return None
    return major, ".{}.{}".format(minor, patch)


def _candidate_cudnn_lib_dirs():
    dirs = []

    for entry in sys.path:
        if not entry:
            continue
        dirs.append(Path(entry) / "nvidia" / "cudnn" / "lib")

    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not entry:
            continue
        path = Path(entry)
        if path.name == "lib" and path.parent.name == "cudnn" and path.parent.parent.name == "nvidia":
            dirs.append(path)

    seen = set()
    for path in dirs:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_dir():
            yield path


def _ensure_cudnn_versioned_aliases(lib_dir, major, suffix):
    created = []
    for lib in sorted(Path(lib_dir).glob("libcudnn*.so.{}".format(major))):
        alias = lib.with_name(lib.name + suffix)
        if alias.exists():
            continue
        try:
            alias.symlink_to(lib.name)
        except FileExistsError:
            continue
        created.append(str(alias))
    return created


def repair_nvidia_cudnn_soname_aliases():
    """Create exact-version cuDNN aliases expected by cuDNN's dlopen path.

    NVIDIA's `nvidia-cudnn-cu13` wheel installs split cuDNN libraries as
    `libcudnn_*.so.9`, but cuDNN 9.22 dlopens exact names such as
    `libcudnn_ops.so.9.22.0`. Without these aliases, the dynamic linker can
    fall through to a system cuDNN installation and mix two cuDNN builds in
    one process, which surfaces as `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`.
    """
    if os.environ.get("MXNET_CUDNN_ALIAS_REPAIR", "1").lower() in ("0", "false", "off", "no"):
        return []

    version = _cudnn_package_version()
    if not version:
        return []
    parsed = _cudnn_alias_suffix(version)
    if not parsed:
        return []
    major, suffix = parsed

    created = []
    for lib_dir in _candidate_cudnn_lib_dirs():
        try:
            created.extend(_ensure_cudnn_versioned_aliases(lib_dir, major, suffix))
        except OSError:
            continue
    return created
