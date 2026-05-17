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
pip-installed nvidia-*-cu13 wheels — PyTorch / JAX install layout.

We do NOT bundle cuDNN / NCCL / CUDA runtime into the wheel:
- Bundling pushes the wheel past the GitHub Releases 2 GB per-asset limit.
- It also makes the wheel huge for users who already have CUDA installed.

Instead we declare them as pip dependencies in setup.py (install_requires)
and patch RUNPATH on libmxnet.so so the loader can find them under
site-packages/nvidia/<pkg>/lib/.

After running this script:
  - python/mxnet/lib/ is gone (no bundled libs)
  - python/mxnet/libmxnet.so has RUNPATH pointing at $ORIGIN/lib + each
    nvidia/<pkg>/lib relative to site-packages/mxnet/

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
    ("nvidia/cudnn/lib", ["libcudnn.so.9"]),
    ("nvidia/nccl/lib", ["libnccl.so.2"]),
]

# CUDA toolkit runtime libs that don't have pip wheels yet — fall back
# to the system install. Order matters; loader tries each in turn.
SYSTEM_CUDA_PATHS = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-13/lib64",
    "/usr/lib/x86_64-linux-gnu",
]


def build_runpath() -> str:
    parts = ["$ORIGIN/lib"]
    parts.extend(f"$ORIGIN/../{d}" for d, _ in NVIDIA_PIP_DEPS)
    parts.extend(SYSTEM_CUDA_PATHS)
    return ":".join(parts)


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
        help="remove python/mxnet/lib/ if it exists (we no longer bundle)",
    )
    args = parser.parse_args()

    libmxnet = Path(args.libmxnet).resolve()
    if not libmxnet.exists():
        print(f"error: {libmxnet} not found", file=sys.stderr)
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

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
