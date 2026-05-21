#!/usr/bin/env python3
#
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

"""Build a repo-local LLVM OpenMP runtime for MXNet.

Invoke through uv so CMake/Ninja and their caches stay under the repository:

  UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
    uv run --with cmake --with ninja python tools/dependencies/build_openmp.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_utils import expected_sha256_for_url, verify_archive_if_pinned, verify_sha256


DEFAULT_VERSION = "22.1.5"
DEFAULT_DEPLOYMENT_TARGET = "11.0"
SAFE_PATH_ENV = "MXNET_OPENMP_SAFE_PATH"


def repo_root() -> Path:
    return Path(__file__).absolute().parents[2]


def absolute(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path


def current_platform_id() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Linux":
        return "linux"
    return system.lower() or "unknown"


def maybe_reexec_from_safe_path() -> None:
    root = repo_root()
    if os.environ.get(SAFE_PATH_ENV) or not any(ch in str(root) for ch in " ()"):
        return

    digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:10]
    safe_root = Path("/private/tmp") / f"mxnet-openmp-{digest}"
    if safe_root.exists():
        if not safe_root.is_symlink() or os.path.realpath(safe_root) != os.path.realpath(root):
            raise SystemExit(f"Safe OpenMP build path already exists and is not this repo: {safe_root}")
    else:
        safe_root.symlink_to(root, target_is_directory=True)

    env = os.environ.copy()
    env[SAFE_PATH_ENV] = "1"
    script = safe_root / "tools" / "dependencies" / "build_openmp.py"
    os.execvpe(sys.executable, [sys.executable, str(script), *sys.argv[1:]], env)


def run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def download(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    retries: int = 5,
    timeout: int = 120,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{retries})", flush=True)
            with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            if expected_sha256:
                try:
                    verify_sha256(tmp, expected_sha256, url)
                except SystemExit:
                    tmp.unlink(missing_ok=True)
                    raise
            tmp.replace(dest)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def _ensure_relative_to(path: Path, dest: Path, member_name: str) -> None:
    try:
        path.relative_to(dest)
    except ValueError:
        raise SystemExit(f"Unsafe archive member path: {member_name}")


def _validate_tar_members(tf: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
    dest = dest.resolve()
    members = tf.getmembers()
    for member in members:
        target = (dest / member.name).resolve()
        _ensure_relative_to(target, dest, member.name)

        if member.islnk():
            link_target = (dest / member.linkname).resolve()
            _ensure_relative_to(link_target, dest, member.name)
        elif member.issym():
            link_target = (target.parent / member.linkname).resolve()
            _ensure_relative_to(link_target, dest, member.name)
        elif not (member.isdir() or member.isreg()):
            raise SystemExit(f"Unsafe archive member type: {member.name}")

    return members


def extract(archive: Path, source_dir: Path) -> None:
    if source_dir.exists():
        return
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {source_dir.parent}", flush=True)
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(source_dir.parent, filter="data")
        except TypeError:
            tf.extractall(source_dir.parent, members=_validate_tar_members(tf, source_dir.parent))


def main() -> int:
    maybe_reexec_from_safe_path()

    root = repo_root()
    arch = platform.machine() or "arm64"
    if arch == "aarch64":
        arch = "arm64"
    system = platform.system()
    platform_id = current_platform_id()

    parser = argparse.ArgumentParser(description="Build repo-local LLVM OpenMP runtime for MXNet.")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--deps-dir", type=Path, default=root / ".deps")
    parser.add_argument("--prefix", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--arch", default=arch)
    parser.add_argument("--deployment-target", default=DEFAULT_DEPLOYMENT_TARGET)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 4, 8))
    args = parser.parse_args()

    deps_dir = absolute(args.deps_dir)
    prefix = (
        absolute(args.prefix)
        if args.prefix
        else deps_dir / f"openmp-{args.version}-{platform_id}-{args.arch}"
    )
    build_dir = (
        absolute(args.build_dir)
        if args.build_dir
        else deps_dir / "build" / f"openmp-{args.version}-{platform_id}-{args.arch}-uv"
    )
    source_dir = deps_dir / "src" / f"llvm-project-{args.version}.src"
    archive = deps_dir / "downloads" / f"llvm-project-{args.version}.src.tar.xz"
    url = (
        "https://github.com/llvm/llvm-project/releases/download/"
        f"llvmorg-{args.version}/llvm-project-{args.version}.src.tar.xz"
    )

    if system == "Darwin" and args.arch not in {"arm64", "x86_64"}:
        raise SystemExit(f"Unsupported macOS architecture: {args.arch}")

    expected_sha256 = expected_sha256_for_url(url)
    if archive.exists():
        verify_archive_if_pinned(archive, url)
    else:
        download(url, archive, expected_sha256=expected_sha256)
    extract(archive, source_dir)

    openmp_source = source_dir / "openmp"
    if not openmp_source.exists():
        raise SystemExit(f"OpenMP source directory not found: {openmp_source}")

    build_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CFLAGS"] = "-fPIC"
    env["CXXFLAGS"] = "-fPIC"

    cmake_args = [
        "cmake",
        "-S",
        str(openmp_source),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
        "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
        "-DOPENMP_ENABLE_LIBOMPTARGET=OFF",
        "-DLIBOMP_ENABLE_SHARED=ON",
        "-DLIBOMP_OMPT_SUPPORT=OFF",
        "-DLIBOMP_USE_HWLOC=OFF",
    ]
    if system == "Darwin":
        cmake_args.extend([
            f"-DCMAKE_OSX_ARCHITECTURES={args.arch}",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={args.deployment_target}",
        ])

    run(cmake_args, env=env)
    run(["cmake", "--build", str(build_dir), "--target", "install", "--parallel", str(args.jobs)])

    print("\nOpenMP runtime installed.")
    print(f"  OPENMP_ROOT={prefix}")
    print(f"  Configure MXNet with -DUSE_OPENMP=ON -DOPENMP_ROOT={prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
