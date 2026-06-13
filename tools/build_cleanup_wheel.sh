#!/usr/bin/env bash
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
#
# Build the cleaned-up Ampere-through-Blackwell Linux/CUDA wheel.
#
# Re-uses the already-configured build/ directory, but pins the CUDA release
# wheel feature set (CUDA/cuDNN/NCCL/oneDNN/OpenCV on,
# sm_80/86/89/90/100/120+PTX) and refreshes CMake metadata first so the binary
# commit stamp matches the current checkout.
# Stages libmxnet.so + libopencv_*.so into python/mxnet/, patches the
# RUNPATH so the loader finds the bundled OpenCV at $ORIGIN/lib, and invokes
# setup.py with MXNET_PACKAGE_VERSION so the wheel metadata matches the
# intended tag.  Runs release_provenance.py at the end and exits non-zero on
# any failure.
#
# Usage:
#   tools/build_cleanup_wheel.sh [<version>]
#
# Default version derives from today: 2.0.0+cu13.bw.YYYYMMDD.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_VERSION="2.0.0+cu13.bw.$(date -u +%Y%m%d)"
VERSION="${1:-${MXNET_PACKAGE_VERSION:-$DEFAULT_VERSION}}"
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="$PYTHON"
elif [ -x "$REPO_ROOT/.venv-mxnet/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv-mxnet/bin/python"
else
    PYTHON_BIN="python3"
fi

# Toggle OpenCV bundling.  When the build was configured with USE_OPENCV=ON
# we copy the system libopencv_*.so files into python/mxnet/lib/ and patch
# libmxnet.so's RUNPATH so the loader finds them next to it.  Set to 0 to
# force an OpenCV-off build (the loader will then complain about missing
# libopencv_* SONAMEs at import time on hosts without system OpenCV).
BUNDLE_OPENCV="${BUNDLE_OPENCV:-1}"

echo "==> Repo: $REPO_ROOT"
echo "==> Version: $VERSION"
echo "==> BUNDLE_OPENCV: $BUNDLE_OPENCV"

if [ ! -f build/CMakeCache.txt ]; then
    echo "build/CMakeCache.txt missing — configure build/ with CMake first" >&2
    exit 2
fi

jobs="${MXNET_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 64)}"
echo "==> Refreshing CMake metadata"
# Explicit release arch set so the wheel always covers Ampere (sm_80/86),
# Ada (sm_89, the validation host), Hopper (sm_90) and Blackwell (sm_100/sm_120
# + PTX) regardless of any cached value. Requires CUDA >= 12.8 for sm_100/120
# (the fork targets CUDA 13); CMakeLists FATALs if an arch yields no gencode (H4).
RELEASE_CUDA_ARCH="${MXNET_CUDA_ARCH:-8.0;8.6;8.9;9.0;10.0;12.0+PTX}"
echo "==> MXNET_CUDA_ARCH: $RELEASE_CUDA_ARCH"
cmake -S . -B build \
    -DUSE_CUDA=ON \
    -DUSE_CUDNN=ON \
    -DUSE_NCCL=ON \
    -DUSE_ONEDNN=ON \
    -DUSE_OPENCV=ON \
    -DMXNET_CUDA_ARCH="$RELEASE_CUDA_ARCH"
echo "==> Building libmxnet.so with $jobs jobs"
cmake --build build --target mxnet --parallel "$jobs"

if [ ! -f build/libmxnet.so ]; then
    echo "build/libmxnet.so missing after build" >&2
    exit 2
fi

# Probe the CMake cache so we know whether OpenCV was actually built in.
HAS_OPENCV=0
if grep -q "USE_OPENCV:BOOL=ON" build/CMakeCache.txt 2>/dev/null; then
    HAS_OPENCV=1
fi
echo "==> CMake USE_OPENCV: $([ "$HAS_OPENCV" = 1 ] && echo ON || echo OFF)"

echo "==> Staging libmxnet.so into python/mxnet/"
cp -v build/libmxnet.so python/mxnet/libmxnet.so

# Stage C/C++ headers into python/mxnet/include/ so the wheel ships them and
# libinfo.find_include_path() resolves to a real in-package directory
# (apache/mxnet#20936). We bundle the standard custom-operator header set
# (mxnet + the nnvm/dlpack/dmlc/mshadow deps that mxnet/*.h include); these are
# exactly the include roots setup.py's Cython extension compiles against.
echo "==> Staging C/C++ headers into python/mxnet/include/"
rm -rf python/mxnet/include
mkdir -p python/mxnet/include
# include/{nnvm,dlpack,dmlc,mshadow} are symlinks into 3rdparty/; use cp -rL to
# dereference them so the real header files (not broken relative symlinks) land
# in the wheel.
for hdr in mxnet nnvm dlpack dmlc mshadow; do
    if [ -e "include/$hdr" ]; then
        cp -rL "include/$hdr" "python/mxnet/include/$hdr"
    else
        echo "  WARNING: include/$hdr not found; skipping" >&2
    fi
done
if [ ! -f python/mxnet/include/mxnet/base.h ]; then
    echo "ERROR: header staging failed; python/mxnet/include/mxnet/base.h missing" >&2
    exit 6
fi

# Reset any prior staging so we always end up with a deterministic lib/
rm -rf python/mxnet/lib

if [ "$HAS_OPENCV" = 1 ] && [ "$BUNDLE_OPENCV" = 1 ]; then
    echo "==> Bundling system OpenCV shared libraries"
    mkdir -p python/mxnet/lib
    # Resolve the OpenCV SONAMEs that libmxnet.so actually depends on.
    needed=$(readelf -d build/libmxnet.so \
        | awk '/NEEDED/ && /libopencv_/ { gsub(/[][]/, "", $5); print $5 }')
    if [ -z "$needed" ]; then
        echo "ERROR: libmxnet.so has no libopencv_ NEEDED entries despite USE_OPENCV=ON" >&2
        exit 5
    fi
    for soname in $needed; do
        # Find the system file the dynamic linker would resolve.  Avoid
        # `awk ... { print; exit }` here — exiting early gives ldconfig a
        # SIGPIPE that pipefail surfaces as a non-zero status.
        target=$(ldconfig -p 2>/dev/null | awk -v s="$soname" '!found && $1 == s { print $NF; found=1 }')
        if [ -z "$target" ] || [ ! -e "$target" ]; then
            # Fall back to scanning the multiarch lib dir directly.
            target=$(find /usr/lib/x86_64-linux-gnu -maxdepth 1 -name "$soname" -print -quit 2>/dev/null)
        fi
        if [ -z "$target" ] || [ ! -e "$target" ]; then
            echo "  FAILED to locate $soname (need libopencv-dev installed)" >&2
            exit 4
        fi
        real=$(readlink -f "$target")
        cp -v "$real" "python/mxnet/lib/$(basename "$real")"
        # Recreate the SONAME symlink that the dynamic linker will look up.
        (cd python/mxnet/lib && ln -sf "$(basename "$real")" "$soname")
    done
    # Also copy direct transitive deps of libopencv that are themselves not in
    # the standard search path (rare on Ubuntu; the OpenCV core libs are
    # self-contained but we still capture any libopencv_imgproc, _core, etc
    # references that may have been pulled in indirectly).
    while :; do
        added=0
        for lib in python/mxnet/lib/libopencv_*.so.*; do
            [ -e "$lib" ] || continue
            for soname in $(readelf -d "$lib" 2>/dev/null \
                | awk '/NEEDED/ && /libopencv_/ { gsub(/[][]/, "", $5); print $5 }'); do
                [ -e "python/mxnet/lib/$soname" ] && continue
                target=$(ldconfig -p 2>/dev/null | awk -v s="$soname" '!found && $1 == s { print $NF; found=1 }')
                [ -z "$target" ] && \
                    target=$(find /usr/lib/x86_64-linux-gnu -maxdepth 1 -name "$soname" -print -quit 2>/dev/null)
                [ -z "$target" ] && continue
                real=$(readlink -f "$target")
                cp -v "$real" "python/mxnet/lib/$(basename "$real")"
                (cd python/mxnet/lib && ln -sf "$(basename "$real")" "$soname")
                added=1
            done
        done
        [ "$added" = 0 ] && break
    done
fi

echo "==> Patching libmxnet.so RUNPATH to include bundled and pip CUDA libraries"
old_runpath=$(patchelf --print-rpath python/mxnet/libmxnet.so || echo "")
new_runpath='$ORIGIN/lib:$ORIGIN/../scipy_openblas32/lib:$ORIGIN/../nvidia/cudnn/lib:$ORIGIN/../nvidia/nccl/lib:$ORIGIN/../nvidia/cu13/lib'
if [ -n "$old_runpath" ]; then
    new_runpath="$new_runpath:$old_runpath"
fi
patchelf --set-rpath "$new_runpath" python/mxnet/libmxnet.so
echo "    new RUNPATH: $new_runpath"

# Verify libmxnet declares a dependency on the CUDA driver (libcuda.so.1).
# libmxnet references Driver-API symbols (e.g. cuLaunchKernel) directly, so it
# MUST carry a DT_NEEDED for libcuda.so.1 — otherwise a clean wheel venv aborts
# dlopen with "undefined symbol: cuLaunchKernel" (the dev tree only works
# because the driver is already loaded globally).  This is supplied at link
# time by `CUDA::cuda_driver` in CMakeLists.txt; the driver itself is
# host-provided by the NVIDIA kernel driver and never bundled.  We do NOT use
# `patchelf --add-needed` for this: it corrupts this ~1 GB binary (the .so loads
# but segfaults during init).  Fail loudly if the NEEDED is missing so the build
# is fixed at the source (link CUDA::cuda_driver) rather than papered over.
if readelf -d python/mxnet/libmxnet.so 2>/dev/null \
        | grep -qiE 'NEEDED.*\blibcuda\.so\.1\b'; then
    echo "==> libcuda.so.1 is a NEEDED dependency (from CUDA::cuda_driver link) — OK"
else
    echo "  FAILED: libmxnet.so has no DT_NEEDED for libcuda.so.1." >&2
    echo "          Ensure CMakeLists.txt links CUDA::cuda_driver and rebuild." >&2
    exit 5
fi

OPENCV_DEPS_FLAG="${OPENCV_DEPS_FLAG:-1}"
if [ "$HAS_OPENCV" != 1 ]; then
    OPENCV_DEPS_FLAG=0
fi

echo "==> Building wheel"
rm -rf dist build_wheel
mkdir -p dist
# MXNET_WITH_CYTHON=1 compiles the _cy3 fast path (setup.py honors the env var
# since `python -m build` does not forward --with-cython). Without it the wheel
# ships ctypes-only and every imperative op pays the slow marshaling cost (M15).
# The cython NDArrayBase now has a __del__ finalizer (PEP 442) matching the
# ctypes class, so NDArrays in reference cycles free their handle on cyclic-GC
# collection (no leak). Requires Cython in the (no-isolation) build env;
# config_cython() degrades to ctypes if Cython is absent.
(cd python && \
    MXNET_PACKAGE_VERSION="$VERSION" \
    MXNET_SETUP_EXCLUDE_ONNX=1 \
    MXNET_SETUP_ENABLE_OPENCV_DEPS="$OPENCV_DEPS_FLAG" \
    MXNET_SETUP_ENABLE_CUDA_DEPS=1 \
    MXNET_WITH_CYTHON=1 \
    "$PYTHON_BIN" -m build --wheel --no-isolation --outdir ../dist)

# Pick the most recently built wheel (-1t), not the lexically-first (-1): if a
# stale wheel from a prior version is still in dist/, lexical order could select
# it and we would validate/ship the wrong artifact (H17).
WHEEL=$(ls -1t dist/*.whl 2>/dev/null | head -n1)
if [ -z "$WHEEL" ]; then
    echo "No wheel produced" >&2
    exit 3
fi
echo "==> Built: $WHEEL"
ls -lh "$WHEEL"

echo "==> Validating provenance"
EXPECT_OPENCV=off
[ "$HAS_OPENCV" = 1 ] && [ "$BUNDLE_OPENCV" = 1 ] && EXPECT_OPENCV=on
"$PYTHON_BIN" tools/release_provenance.py "$WHEEL" \
    --cmake-cache build/CMakeCache.txt \
    --package-version "$VERSION" \
    --expect-cuda on \
    --expect-cudnn on \
    --expect-nccl on \
    --expect-onednn on \
    --expect-opencv "$EXPECT_OPENCV"

echo "==> Wheel build OK: $WHEEL"
