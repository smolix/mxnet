#!/usr/bin/env python3
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
"""Set libmxnet.so RUNPATH so the wheel finds NVIDIA runtime libs via
pip-installed nvidia-*-cu13 wheels - PyTorch / JAX install layout.

We do NOT bundle cuDNN / NCCL / CUDA runtime into the wheel:
- Bundling pushes the wheel past the GitHub Releases 2 GB per-asset limit.
- It also makes the wheel huge for users who already have CUDA installed.

Instead we declare them as pip dependencies in setup.py (install_requires)
and patch RUNPATH on libmxnet.so so the loader can find them under
site-packages/nvidia/<pkg>/lib/.

OpenCV is different from the NVIDIA runtime libraries: a system-linked
libmxnet.so needs libopencv_*.so SONAMEs that Python package metadata cannot
install reliably. If libmxnet.so links to OpenCV, either bundle those libraries
with --bundle-opencv, build the wheel with USE_OPENCV=OFF, or deliberately pass
--allow-system-opencv and document the required OS packages.

After running this script with --drop-bundled and without --bundle-opencv:
  - python/mxnet/lib/ is gone (no bundled libs)
  - python/mxnet/libmxnet.so has RUNPATH pointing at $ORIGIN/lib + each
    scipy_openblas32/lib and nvidia/<pkg>/lib relative to site-packages/mxnet/

Requires:
  - patchelf  (apt install patchelf)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# Each entry maps the directory the NVIDIA wheel installs under
# site-packages/ to the libraries we expect to find there. The RUNPATH
# is constructed relative to mxnet/libmxnet.so, i.e. $ORIGIN/.. is
# site-packages.
# NVIDIA wheels we can pip-install for CUDA 13 (as of 2026-05-17 NVIDIA
# only ships these two for cu13 on PyPI; the rest are placeholder 0.0.1
# stubs and so come from the system toolkit at /usr/local/cuda/ for now).
NVIDIA_PIP_DEPS = [
    ("scipy_openblas32/lib", ["libscipy_openblas.so"]),
    ("nvidia/cudnn/lib", ["libcudnn.so.9"]),
    ("nvidia/nccl/lib", ["libnccl.so.2"]),
    ("nvidia/cu13/lib", ["libnvrtc.so.13", "libcublas.so.13", "libcublasLt.so.13"]),
]

# CUDA toolkit runtime libs that don't have pip wheels yet — fall back
# to the system install. Order matters; loader tries each in turn.
SYSTEM_CUDA_PATHS = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-13/lib64",
    "/usr/lib/x86_64-linux-gnu",
]

OPENCV_LIBRARY_PREFIX = "libopencv_"


def build_runpath() -> str:
    parts = ["$ORIGIN/lib"]
    parts.extend(f"$ORIGIN/../{d}" for d, _ in NVIDIA_PIP_DEPS)
    parts.extend(SYSTEM_CUDA_PATHS)
    return ":".join(parts)


def needed_libraries(binary: Path) -> list[str]:
    """Return DT_NEEDED entries for a shared library."""
    if shutil.which("patchelf"):
        result = subprocess.run(
            ["patchelf", "--print-needed", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if shutil.which("readelf"):
        result = subprocess.run(
            ["readelf", "-d", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        needed = []
        for line in result.stdout.splitlines():
            marker = "Shared library: ["
            if marker in line:
                needed.append(line.split(marker, 1)[1].split("]", 1)[0])
        return needed
    raise RuntimeError("patchelf or readelf is required to inspect DT_NEEDED")


def needed_opencv_libraries(binary: Path) -> list[str]:
    """Return direct OpenCV shared-library dependencies for libmxnet."""
    return [
        name
        for name in needed_libraries(binary)
        if name.startswith(OPENCV_LIBRARY_PREFIX)
    ]


def bundled_opencv_libraries(libmxnet: Path) -> list[Path]:
    """Return staged OpenCV libraries that would be included by setup.py."""
    bundled_dir = libmxnet.parent / "lib"
    if not bundled_dir.is_dir():
        return []
    return sorted(bundled_dir.glob(f"{OPENCV_LIBRARY_PREFIX}*.so*"))


def validate_opencv_policy(
    opencv_needed: list[str],
    bundled_opencv: list[str],
    drop_bundled: bool,
    bundle_opencv: bool,
    allow_system_opencv: bool,
) -> None:
    """Reject wheels that would silently require system OpenCV."""
    if not opencv_needed:
        return
    if bundle_opencv or allow_system_opencv:
        return
    if set(opencv_needed).issubset(set(bundled_opencv)) and not drop_bundled:
        return
    libs = ", ".join(opencv_needed)
    raise RuntimeError(
        "libmxnet.so links to OpenCV ({libs}), but this wheel build would not "
        "bundle those libraries. The opencv-python package only covers Python "
        "cv2 imports; it does not satisfy libmxnet.so OpenCV SONAME dependencies. "
        "Rebuild with USE_OPENCV=OFF, pass --bundle-opencv, or pass "
        "--allow-system-opencv and document the required OS packages.".format(libs=libs)
    )


def resolved_library_paths(binary: Path) -> dict[str, Path]:
    """Return shared libraries resolved by ldd, keyed by SONAME."""
    result = subprocess.run(
        ["ldd", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    paths = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if "=>" not in line:
            continue
        name, rest = line.split("=>", 1)
        name = name.strip()
        rest = rest.strip()
        if rest.startswith("not found"):
            continue
        path = rest.split(None, 1)[0]
        if path.startswith("/"):
            paths[name] = Path(path)
    return paths


def bundle_opencv_libraries(libmxnet: Path) -> list[Path]:
    """Copy resolved OpenCV runtime libraries next to libmxnet."""
    opencv_needed = needed_opencv_libraries(libmxnet)
    if not opencv_needed:
        return []

    resolved = resolved_library_paths(libmxnet)
    opencv_paths = {
        name: path
        for name, path in resolved.items()
        if name.startswith(OPENCV_LIBRARY_PREFIX)
    }
    missing = [name for name in opencv_needed if name not in opencv_paths]
    if missing:
        raise RuntimeError(
            "OpenCV libraries were not resolved by ldd: "
            + ", ".join(sorted(missing))
        )

    bundled_dir = libmxnet.parent / "lib"
    bundled_dir.mkdir(exist_ok=True)
    copied = []
    for name, source in sorted(opencv_paths.items()):
        dest = bundled_dir / name
        shutil.copy2(source, dest)
        copied.append(dest)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libmxnet",
        default=str(Path(__file__).resolve().parents[1] / "mxnet" / "libmxnet.so"),
        help="path to libmxnet.so to patch (default: python/mxnet/libmxnet.so)",
    )
    parser.add_argument(
        "--drop-bundled",
        action="store_true",
        help="remove python/mxnet/lib/ before optional OpenCV rebundling",
    )
    parser.add_argument(
        "--bundle-opencv",
        action="store_true",
        help="copy resolved libopencv_*.so* dependencies into python/mxnet/lib/",
    )
    parser.add_argument(
        "--allow-system-opencv",
        action="store_true",
        help="allow a wheel that requires system OpenCV packages",
    )
    args = parser.parse_args()

    libmxnet = Path(args.libmxnet).resolve()
    if not libmxnet.exists():
        print(f"error: {libmxnet} not found", file=sys.stderr)
        return 1

    try:
        opencv_needed = needed_opencv_libraries(libmxnet)
        validate_opencv_policy(
            opencv_needed=opencv_needed,
            bundled_opencv=[path.name for path in bundled_opencv_libraries(libmxnet)],
            drop_bundled=args.drop_bundled,
            bundle_opencv=args.bundle_opencv,
            allow_system_opencv=args.allow_system_opencv,
        )
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    runpath = build_runpath()
    print(f"patching {libmxnet}")
    print(f"  RUNPATH={runpath}")
    subprocess.run(
        ["patchelf", "--set-rpath", runpath, str(libmxnet)],
        check=True,
    )

    actual = subprocess.run(
        ["patchelf", "--print-rpath", str(libmxnet)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual != runpath:
        print(
            f"error: RUNPATH didn't take. got: {actual!r}, want: {runpath!r}",
            file=sys.stderr,
        )
        return 1

    if args.drop_bundled:
        bundled = libmxnet.parent / "lib"
        if bundled.is_dir():
            print(f"removing {bundled}/")
            shutil.rmtree(bundled)

    if args.bundle_opencv:
        try:
            copied = bundle_opencv_libraries(libmxnet)
        except RuntimeError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        if copied:
            print("bundled OpenCV libraries:")
            for path in copied:
                print(f"  {path}")
        else:
            print("OpenCV is not linked by libmxnet.so; nothing to bundle.")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
