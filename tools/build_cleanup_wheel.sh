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
# Build a self-contained MXNet wheel and bundle the OpenCV transitive closure.
#
# Two host platforms are supported, auto-detected from `uname -s`:
#
#   * Linux  — the Ampere-through-Blackwell CUDA 13 wheel.  Pins the CUDA
#              release feature set (CUDA/cuDNN/NCCL/oneDNN/OpenCV on,
#              sm_80/86/89/90/100/120+PTX), stages libmxnet.so, bundles the
#              OpenCV closure with patchelf/$ORIGIN, and depends on the pip
#              nvidia-*-cu13 packages for the CUDA runtime.
#
#   * macOS  — the Apple-silicon CPU wheel.  CPU-only (no CUDA/cuDNN/NCCL),
#              Accelerate BLAS/LAPACK, oneDNN on, OpenCV on, stages
#              libmxnet.dylib and bundles the OpenCV closure with
#              install_name_tool/@loader_path (the Mach-O analog of the Linux
#              $ORIGIN fix).
#
# In both cases the OpenCV bundling vendors the FULL transitive closure of the
# native OpenCV libraries — not just their libopencv_* siblings — so the wheel
# imports on a clean host with no system OpenCV (and no system codec/geo stack).
# Each vendored library is repointed at its siblings inside mxnet/lib/ via the
# loader's "relative to the loading object" mechanism ($ORIGIN on ELF,
# @loader_path on Mach-O), which — unlike an inherited RUNPATH/LC_RPATH — is
# honored per library and so survives deep dependency chains.
#
# Runs release_provenance.py at the end and exits non-zero on any failure.
#
# Usage:
#   tools/build_cleanup_wheel.sh [<version>]
#
# Default version derives from today and the host:
#   Linux:  2.0.0+cu13.bw.YYYYMMDD
#   macOS:  2.0.0+cpu.macos.YYYYMMDD
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OS="$(uname -s)"

# Wheel flavor selects the feature set and the build tree:
#   macos      — Apple-silicon CPU wheel (Darwin)
#   linux-cuda — Ampere→Blackwell CUDA 13 wheel (Linux default)
#   linux-cpu  — x86_64 CPU wheel (Linux, MXNET_WHEEL_FLAVOR=cpu): no CUDA stack,
#                oneDNN + OpenCV on, OpenCV bundled exactly like the other flavors.
# The CPU wheel uses a SEPARATE build tree (build-cpu/) so it never clobbers — and
# is never clobbered by — the CUDA build/ (override with MXNET_BUILD_DIR).
case "$OS" in
    Darwin) FLAVOR=macos ;;
    *)
        case "${MXNET_WHEEL_FLAVOR:-cuda}" in
            cpu)  FLAVOR=linux-cpu ;;
            cuda) FLAVOR=linux-cuda ;;
            *) echo "unknown MXNET_WHEEL_FLAVOR='${MXNET_WHEEL_FLAVOR}' (use 'cuda' or 'cpu')" >&2; exit 2 ;;
        esac
        ;;
esac
case "$FLAVOR" in
    linux-cpu) BUILD_DIR="${MXNET_BUILD_DIR:-build-cpu}" ;;
    *)         BUILD_DIR="${MXNET_BUILD_DIR:-build}" ;;
esac

case "$FLAVOR" in
    macos)     DEFAULT_VERSION="2.0.0+cpu.macos.$(date -u +%Y%m%d)" ;;
    linux-cpu) DEFAULT_VERSION="2.0.0+cpu.linux.$(date -u +%Y%m%d)" ;;
    *)         DEFAULT_VERSION="2.0.0+cu13.bw.$(date -u +%Y%m%d)" ;;
esac
VERSION="${1:-${MXNET_PACKAGE_VERSION:-$DEFAULT_VERSION}}"
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="$PYTHON"
elif [ -x "$REPO_ROOT/.venv-mxnet/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv-mxnet/bin/python"
else
    PYTHON_BIN="python3"
fi

# Toggle OpenCV bundling.  When the build was configured with USE_OPENCV=ON we
# copy the system OpenCV shared libraries (and their full transitive closure)
# into python/mxnet/lib/ and repoint libmxnet so the loader finds them next to
# it.  Set to 0 to force an OpenCV-off wheel (the loader will then complain about
# missing OpenCV at import time on hosts without system OpenCV).
BUNDLE_OPENCV="${BUNDLE_OPENCV:-1}"

echo "==> Repo: $REPO_ROOT"
echo "==> Host OS: $OS"
echo "==> Flavor: $FLAVOR"
echo "==> Build dir: $BUILD_DIR"
echo "==> Version: $VERSION"
echo "==> BUNDLE_OPENCV: $BUNDLE_OPENCV"

# linux-cpu and macos both do a COMPLETE first-time configure below (generator,
# build type, arch, BLAS, all feature toggles), so they may start from a clean
# tree.  Only linux-cuda reuses a pre-seeded cache: its reconfigure deliberately
# does not set the generator/build-type/BLAS (those come from the §3 first-time
# configure in docs/cuda_wheel_build.md), so it requires the cache to exist.
if [ "$FLAVOR" = linux-cuda ] && [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "$BUILD_DIR/CMakeCache.txt missing — configure $BUILD_DIR/ with CMake first" >&2
    echo "(see docs/cuda_wheel_build.md §3 'First-time configure')" >&2
    exit 2
fi

jobs="${MXNET_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 64)}"

# ----------------------------------------------------------------------
# OpenCV bundling — Linux (ELF / patchelf / $ORIGIN)
# ----------------------------------------------------------------------
bundle_opencv_closure_linux() {
    echo "==> Bundling system OpenCV shared libraries (Linux/ELF)"
    mkdir -p python/mxnet/lib
    # Resolve the OpenCV SONAMEs that libmxnet.so actually depends on.
    local needed
    needed=$(readelf -d "$BUILD_DIR/libmxnet.so" \
        | awk '/NEEDED/ && /libopencv_/ { gsub(/[][]/, "", $5); print $5 }')
    if [ -z "$needed" ]; then
        echo "ERROR: libmxnet.so has no libopencv_ NEEDED entries despite USE_OPENCV=ON" >&2
        exit 5
    fi
    local soname target real
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
    # Vendor the FULL transitive closure of the bundled OpenCV libraries — not
    # just their libopencv_* siblings. Ubuntu's libopencv_imgcodecs links a deep
    # codec/geo stack (libgdcm*, libgdal, libOpenEXR*/libImath, libtbb,
    # libjpeg/png/tiff/webp/openjp2, …). Bundling only the libopencv_* files (the
    # old behaviour, which filtered NEEDED on /libopencv_/) left those out, so a
    # clean host without them failed at `import mxnet` with e.g.
    #   OSError: libgdcmMSFF.so.3.0: cannot open shared object file
    # Walk every NEEDED soname reachable from the already-bundled libs to a
    # fixpoint and copy each, EXCEPT:
    #   * the C runtime / toolchain (glibc, libstdc++, libgcc_s, ld-linux) — the
    #     host ABI, which must never be shipped; and
    #   * CUDA runtime / driver libs — intentionally host- or pip-provided and
    #     resolved via libmxnet's RUNPATH (libcuda, libnvidia-*, libcudart,
    #     libcublas, libcudnn, libnccl, libnvrtc, libcu{fft,solver,sparse,rand}).
    #
    # Each name is anchored on a soname boundary — (\.so|-|$) — so it matches a WHOLE
    # library name, never a prefix. Without the anchor the short glibc names match
    # unrelated libraries: `libc` matches libcharls.so.2 / libcrypto / libcurl, `libm`
    # matches libmount, `librt` matches librtmp. That silently dropped real OpenCV
    # codec deps from the closure, so a clean host (no system OpenCV) failed at
    # `import mxnet` with e.g. "OSError: libcharls.so.2: cannot open shared object file".
    local skip_re='^(ld-linux|libc|libm|libdl|libpthread|librt|libutil|libnsl|libresolv|libstdc\+\+|libgcc_s|libcuda|libcudart|libcublas|libcudnn|libnccl|libnvrtc|libcufft|libcusolver|libcusparse|libcurand|libnvJitLink|libnvToolsExt|libnvidia)(\.so|-|$)'
    local added lib real
    while :; do
        added=0
        for lib in python/mxnet/lib/*.so*; do
            [ -f "$lib" ] || continue   # skip the SONAME symlinks
            for soname in $(readelf -d "$lib" 2>/dev/null \
                | awk '/NEEDED/ { gsub(/[][]/, "", $5); print $5 }'); do
                printf '%s\n' "$soname" | grep -qE "$skip_re" && continue
                [ -e "python/mxnet/lib/$soname" ] && continue
                target=$(ldconfig -p 2>/dev/null | awk -v s="$soname" '!found && $1 == s { print $NF; found=1 }')
                if [ -z "$target" ] || [ ! -e "$target" ]; then
                    target=$(find /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu \
                        -maxdepth 1 -name "$soname" -print -quit 2>/dev/null)
                fi
                if [ -z "$target" ] || [ ! -e "$target" ]; then
                    echo "  WARNING: unresolved transitive dep $soname (left to host)" >&2
                    continue
                fi
                real=$(readlink -f "$target")
                cp -v "$real" "python/mxnet/lib/$(basename "$real")"
                # Recreate the SONAME symlink unless the file already IS the
                # soname (e.g. libdeflate.so.0) — `ln -sf X X` would self-link.
                if [ "$(basename "$real")" != "$soname" ]; then
                    (cd python/mxnet/lib && ln -sf "$(basename "$real")" "$soname")
                fi
                added=1
            done
        done
        [ "$added" = 0 ] && break
    done
    # Point every bundled library at its siblings: DT_RUNPATH is not inherited by
    # a library's own dependencies, so each vendored .so must carry $ORIGIN to
    # find the others in python/mxnet/lib/ at load time.
    for lib in python/mxnet/lib/*.so*; do
        [ -f "$lib" ] || continue   # real files only, not the SONAME symlinks
        patchelf --set-rpath '$ORIGIN' "$lib" 2>/dev/null || true
    done
    echo "==> Bundled OpenCV closure: $(find python/mxnet/lib -maxdepth 1 -type f -name '*.so*' | wc -l) shared objects in python/mxnet/lib/"
}

# ----------------------------------------------------------------------
# OpenCV bundling — macOS (Mach-O / install_name_tool / @loader_path)
# ----------------------------------------------------------------------
# A dependency install-name is host-provided (never bundled) when it lives in a
# macOS system prefix, or is already a loader-relative @-path we cannot resolve
# to a concrete file on disk.
_macos_is_system_dylib() {
    case "$1" in
        /usr/lib/*|/System/*) return 0 ;;
        @rpath/*|@loader_path/*|@executable_path/*) return 0 ;;
        *) return 1 ;;
    esac
}

# The library's own install id (LC_ID_DYLIB), or empty for a non-dylib.
_macos_dylib_id() {
    otool -D "$1" 2>/dev/null | tail -n +2 | head -n1
}

# The Mach-O dependencies (LC_LOAD_DYLIB) of a binary, excluding its own id.
# `otool -L` prints "<path>:" then the id then one dependency per line; strip
# the header line and the self-id so only real dependencies remain.
_macos_dylib_deps() {
    local self_id
    self_id="$(_macos_dylib_id "$1")"
    otool -L "$1" 2>/dev/null | tail -n +2 | awk '{print $1}' \
        | while read -r dep; do
            [ -z "$dep" ] && continue
            [ "$dep" = "$self_id" ] && continue
            printf '%s\n' "$dep"
        done
}

bundle_opencv_closure_macos() {
    echo "==> Bundling system OpenCV shared libraries (macOS/Mach-O)"
    mkdir -p python/mxnet/lib
    local dep b seeded=0
    # Seed: the OpenCV dylibs libmxnet.dylib links directly.
    for dep in $(_macos_dylib_deps "$BUILD_DIR/libmxnet.dylib"); do
        case "$(basename "$dep")" in
            libopencv_*) ;;
            *) continue ;;
        esac
        _macos_is_system_dylib "$dep" && continue
        b="$(basename "$dep")"
        if [ ! -e "$dep" ]; then
            echo "  FAILED to locate OpenCV dep $dep referenced by libmxnet.dylib" >&2
            exit 4
        fi
        cp -L "$dep" "python/mxnet/lib/$b"
        chmod u+w "python/mxnet/lib/$b"
        echo "  bundled (seed) $b"
        seeded=1
    done
    if [ "$seeded" = 0 ]; then
        echo "ERROR: $BUILD_DIR/libmxnet.dylib has no libopencv_* dependencies despite USE_OPENCV=ON" >&2
        exit 5
    fi
    # Walk the FULL transitive closure to a fixpoint: every non-system dylib
    # reachable from an already-bundled lib gets vendored too. OpenCV's
    # imgcodecs pulls a deep codec stack (libjpeg/png/tiff/webp/openjp2,
    # OpenEXR/Imath/Iex/IlmThread, …) — all under the package-manager prefix,
    # none of them macOS-provided — exactly the Mach-O analog of the Linux
    # libgdcm*/libgdal/… closure.
    local added lib
    while :; do
        added=0
        for lib in python/mxnet/lib/*.dylib; do
            [ -e "$lib" ] || continue
            for dep in $(_macos_dylib_deps "$lib"); do
                _macos_is_system_dylib "$dep" && continue
                b="$(basename "$dep")"
                [ -e "python/mxnet/lib/$b" ] && continue
                if [ ! -e "$dep" ]; then
                    echo "  WARNING: unresolved transitive dep $dep (left to host)" >&2
                    continue
                fi
                cp -L "$dep" "python/mxnet/lib/$b"
                chmod u+w "python/mxnet/lib/$b"
                echo "  bundled $b"
                added=1
            done
        done
        [ "$added" = 0 ] && break
    done
    # Repoint every vendored dylib at its siblings.  macOS resolves a dependency
    # recorded as @loader_path/NAME relative to the directory of the binary doing
    # the loading — the exact analog of ELF $ORIGIN — and (unlike an LC_RPATH) it
    # is honored per library, so it is NOT subject to the "RUNPATH not inherited
    # by a library's own deps" problem the Linux fix had to work around.  Every
    # bundled lib lives in the same directory, so each sibling reference becomes
    # @loader_path/<sibling>.  The id is set to @rpath/<name> for cleanliness.
    for lib in python/mxnet/lib/*.dylib; do
        [ -e "$lib" ] || continue
        b="$(basename "$lib")"
        install_name_tool -id "@rpath/$b" "$lib"
        for dep in $(_macos_dylib_deps "$lib"); do
            if [ -e "python/mxnet/lib/$(basename "$dep")" ]; then
                install_name_tool -change "$dep" "@loader_path/$(basename "$dep")" "$lib"
            fi
        done
        # install_name_tool invalidates the (ad-hoc) code signature; re-sign so
        # dyld on Apple Silicon will load the rewritten dylib.
        codesign --force --sign - "$lib" 2>/dev/null || true
    done
    echo "==> Bundled OpenCV closure: $(find python/mxnet/lib -maxdepth 1 -type f -name '*.dylib' | wc -l | tr -d ' ') dylibs in python/mxnet/lib/"
}

# ----------------------------------------------------------------------
# OpenMP runtime bundling — macOS (Mach-O / install_name_tool / @loader_path)
# ----------------------------------------------------------------------
# Unlike Linux — where the OpenMP runtime (libgomp.so.1) is part of the host GCC
# runtime and is intentionally left host-provided — macOS ships NO system libomp.
# So a self-contained wheel must vendor the hermetic libomp.dylib it was linked
# against.  libmxnet records that dependency as @rpath/libomp.dylib (the lib's
# install id); the install-name pass on libmxnet repoints it to
# @loader_path/lib/libomp.dylib once the file is present in mxnet/lib/.
bundle_openmp_runtime_macos() {
    echo "==> Bundling OpenMP runtime (libomp) into the wheel (macOS/Mach-O)"
    mkdir -p python/mxnet/lib
    # Resolve the exact libomp.dylib CMake linked against, straight from the cache.
    local omp_lib dep
    omp_lib="$(awk -F= '/^(MXNET_OPENMP_LIBRARY|OpenMP_omp_LIBRARY):/ {print $2}' \
        "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | tail -n1)"
    if [ -z "$omp_lib" ] || [ ! -e "$omp_lib" ]; then
        omp_lib="${OPENMP_ROOT_HINT:+$OPENMP_ROOT_HINT/lib/libomp.dylib}"
    fi
    if [ -z "$omp_lib" ] || [ ! -e "$omp_lib" ]; then
        echo "ERROR: USE_OPENMP=ON but could not resolve libomp.dylib to bundle" >&2
        exit 5
    fi
    cp -L "$omp_lib" python/mxnet/lib/libomp.dylib
    chmod u+w python/mxnet/lib/libomp.dylib
    install_name_tool -id "@rpath/libomp.dylib" python/mxnet/lib/libomp.dylib
    # libomp's own deps are macOS-system only (libSystem); repoint any that happen
    # to be siblings, for parity with the OpenCV closure handling.
    for dep in $(_macos_dylib_deps python/mxnet/lib/libomp.dylib); do
        if [ -e "python/mxnet/lib/$(basename "$dep")" ]; then
            install_name_tool -change "$dep" "@loader_path/$(basename "$dep")" \
                python/mxnet/lib/libomp.dylib
        fi
    done
    codesign --force --sign - python/mxnet/lib/libomp.dylib 2>/dev/null || true
    echo "    bundled libomp.dylib from $omp_lib"
}

# ----------------------------------------------------------------------
# Configure + build libmxnet
# ----------------------------------------------------------------------
if [ "$FLAVOR" = macos ]; then
    echo "==> Refreshing CMake metadata (macOS arm64 CPU + OpenCV + OpenMP)"
    # CPU-only Apple-silicon feature set: no CUDA stack, Accelerate BLAS/LAPACK,
    # oneDNN on, OpenCV on (so mx.image / RecordIO native decode works), OpenMP
    # ON (parity with the Linux wheels; it also switches oneDNN from the
    # single-threaded SEQ runtime to the multi-threaded OMP runtime), no x86
    # SSE/F16C.  OpenCV is discovered via OpenCV_DIR (override with the OpenCV_DIR
    # env var); the MacPorts default is /opt/local/libexec/opencv4/cmake.
    OPENCV_DIR_HINT="${OpenCV_DIR:-/opt/local/libexec/opencv4/cmake}"
    echo "==> OpenCV_DIR: $OPENCV_DIR_HINT"
    # AppleClang ships no OpenMP runtime and CMakeLists.txt deliberately does NOT
    # probe system/Homebrew/MacPorts libomp: it uses a hermetic libomp built by
    # tools/dependencies/build_openmp.py and installed under .deps/openmp-*.  Build
    # it on demand if absent so USE_OPENMP=ON always has a runtime, then point
    # CMake at it explicitly (the same prefix's libomp.dylib is bundled into the
    # wheel below for self-containment, since there is no system libomp to fall
    # back on the way Linux falls back on the host libgomp).
    OPENMP_ROOT_HINT="${OPENMP_ROOT:-$(ls -d "$REPO_ROOT"/.deps/openmp-*-macos-* 2>/dev/null | head -n1)}"
    if [ -z "$OPENMP_ROOT_HINT" ] || [ ! -f "$OPENMP_ROOT_HINT/include/omp.h" ]; then
        echo "==> No hermetic libomp under .deps/; building it (tools/dependencies/build_openmp.py)"
        "$PYTHON_BIN" tools/dependencies/build_openmp.py
        OPENMP_ROOT_HINT="$(ls -d "$REPO_ROOT"/.deps/openmp-*-macos-* 2>/dev/null | head -n1)"
    fi
    if [ -z "$OPENMP_ROOT_HINT" ] || [ ! -f "$OPENMP_ROOT_HINT/include/omp.h" ]; then
        echo "ERROR: USE_OPENMP=ON on macOS but no hermetic libomp prefix is available" >&2
        exit 2
    fi
    echo "==> OPENMP_ROOT: $OPENMP_ROOT_HINT"
    cmake -S . -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DUSE_CUDA=OFF \
        -DUSE_CUDNN=OFF \
        -DUSE_NCCL=OFF \
        -DUSE_ONEDNN=ON \
        -DUSE_OPENMP=ON \
        -DOPENMP_ROOT="$OPENMP_ROOT_HINT" \
        -DUSE_OPENCV=ON \
        -DOpenCV_DIR="$OPENCV_DIR_HINT" \
        -DUSE_BLAS=apple \
        -DUSE_LAPACK=ON \
        -DUSE_DIST_KVSTORE=OFF \
        -DUSE_SSE=OFF \
        -DUSE_F16C=OFF \
        -DBUILD_CPP_EXAMPLES=OFF
    LIBMXNET_BUILT="$BUILD_DIR/libmxnet.dylib"
    LIBMXNET_STAGED="python/mxnet/libmxnet.dylib"
elif [ "$FLAVOR" = linux-cpu ]; then
    echo "==> Configuring CMake ($BUILD_DIR: Linux x86_64 CPU + oneDNN + OpenCV)"
    # x86_64 CPU feature set: no CUDA stack, OpenBLAS/LAPACK, oneDNN on, OpenCV on
    # (so mx.image / RecordIO native decode works — parity with the CUDA wheel),
    # OpenMP + F16C on.  This is a COMPLETE first-time configure (generator, build
    # type, BLAS all set) so build-cpu/ can be created from a clean tree.
    cmake -S . -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=OFF \
        -DUSE_CUDNN=OFF \
        -DUSE_NCCL=OFF \
        -DUSE_ONEDNN=ON \
        -DUSE_OPENMP=ON \
        -DUSE_F16C=ON \
        -DUSE_OPENCV=ON \
        -DUSE_LAPACK=ON \
        -DUSE_BLAS=open \
        -DUSE_DIST_KVSTORE=OFF \
        -DBUILD_CPP_EXAMPLES=OFF
    LIBMXNET_BUILT="$BUILD_DIR/libmxnet.so"
    LIBMXNET_STAGED="python/mxnet/libmxnet.so"
else
    echo "==> Refreshing CMake metadata (Linux CUDA)"
    # Explicit release arch set so the wheel always covers Ampere (sm_80/86),
    # Ada (sm_89, the validation host), Hopper (sm_90) and Blackwell
    # (sm_100/sm_120 + PTX) regardless of any cached value. Requires CUDA >= 12.8
    # for sm_100/120 (the fork targets CUDA 13); CMakeLists FATALs if an arch
    # yields no gencode (H4).
    RELEASE_CUDA_ARCH="${MXNET_CUDA_ARCH:-8.0;8.6;8.9;9.0;10.0;12.0+PTX}"
    echo "==> MXNET_CUDA_ARCH: $RELEASE_CUDA_ARCH"
    cmake -S . -B "$BUILD_DIR" \
        -DUSE_CUDA=ON \
        -DUSE_CUDNN=ON \
        -DUSE_NCCL=ON \
        -DUSE_ONEDNN=ON \
        -DUSE_OPENCV=ON \
        -DMXNET_CUDA_ARCH="$RELEASE_CUDA_ARCH"
    LIBMXNET_BUILT="$BUILD_DIR/libmxnet.so"
    LIBMXNET_STAGED="python/mxnet/libmxnet.so"
fi

echo "==> Building libmxnet with $jobs jobs"
cmake --build "$BUILD_DIR" --target mxnet --parallel "$jobs"

if [ ! -f "$LIBMXNET_BUILT" ]; then
    echo "$LIBMXNET_BUILT missing after build" >&2
    exit 2
fi

# Probe the CMake cache so we know whether OpenCV was actually built in.
HAS_OPENCV=0
if grep -q "USE_OPENCV:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    HAS_OPENCV=1
fi
echo "==> CMake USE_OPENCV: $([ "$HAS_OPENCV" = 1 ] && echo ON || echo OFF)"

echo "==> Staging $(basename "$LIBMXNET_BUILT") into python/mxnet/"
cp -v "$LIBMXNET_BUILT" "$LIBMXNET_STAGED"

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
    if [ "$OS" = Darwin ]; then
        bundle_opencv_closure_macos
    else
        bundle_opencv_closure_linux
    fi
fi

# Vendor the OpenMP runtime on macOS (no system libomp to fall back on).  Probe
# the cache so this fires exactly when libmxnet was actually built with OpenMP.
if [ "$FLAVOR" = macos ] && grep -q "USE_OPENMP:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    bundle_openmp_runtime_macos
fi

# ----------------------------------------------------------------------
# Repoint libmxnet at the bundled libraries
# ----------------------------------------------------------------------
if [ "$FLAVOR" = macos ]; then
    echo "==> Patching libmxnet.dylib install names to the bundled OpenCV closure"
    # Give libmxnet a relocatable id so the Cython extension (built next, with
    # -rpath @loader_path/..) and any consumer resolve it from inside the package
    # rather than from the absolute build path.
    install_name_tool -id "@rpath/libmxnet.dylib" python/mxnet/libmxnet.dylib
    # Repoint each bundled-OpenCV dependency to the wheel-local lib/ directory.
    # libmxnet.dylib sits one level above lib/, so @loader_path/lib/<name>.
    for dep in $(_macos_dylib_deps python/mxnet/libmxnet.dylib); do
        if [ -e "python/mxnet/lib/$(basename "$dep")" ]; then
            install_name_tool -change "$dep" "@loader_path/lib/$(basename "$dep")" \
                python/mxnet/libmxnet.dylib
        fi
    done
    # Belt-and-suspenders: an LC_RPATH of @loader_path/lib so any residual
    # @rpath/<name> id also resolves into the bundled directory.
    install_name_tool -add_rpath "@loader_path/lib" python/mxnet/libmxnet.dylib 2>/dev/null || true
    codesign --force --sign - python/mxnet/libmxnet.dylib 2>/dev/null || true
    echo "    libmxnet.dylib OpenCV references now:"
    otool -L python/mxnet/libmxnet.dylib | awk '/libopencv_/ {print "      " $1}'
elif [ "$FLAVOR" = linux-cpu ]; then
    echo "==> Patching libmxnet.so RUNPATH (CPU wheel: bundled OpenCV + OpenBLAS)"
    # CPU wheel: no NVIDIA runtime, so the RUNPATH only needs the bundled OpenCV
    # closure ($ORIGIN/lib) and the scipy-openblas32 OpenBLAS.  No libcuda.so.1
    # dependency is expected (USE_CUDA=OFF).
    old_runpath=$(patchelf --print-rpath python/mxnet/libmxnet.so || echo "")
    new_runpath='$ORIGIN/lib:$ORIGIN/../scipy_openblas32/lib'
    if [ -n "$old_runpath" ]; then
        new_runpath="$new_runpath:$old_runpath"
    fi
    patchelf --set-rpath "$new_runpath" python/mxnet/libmxnet.so
    echo "    new RUNPATH: $new_runpath"
    if readelf -d python/mxnet/libmxnet.so 2>/dev/null | grep -qiE 'NEEDED.*\blibcuda'; then
        echo "  FAILED: CPU wheel libmxnet.so unexpectedly links a CUDA library." >&2
        exit 5
    fi
else
    echo "==> Patching libmxnet.so RUNPATH to include bundled and pip CUDA libraries"
    old_runpath=$(patchelf --print-rpath python/mxnet/libmxnet.so || echo "")
    new_runpath='$ORIGIN/lib:$ORIGIN/../scipy_openblas32/lib:$ORIGIN/../nvidia/cudnn/lib:$ORIGIN/../nvidia/nccl/lib:$ORIGIN/../nvidia/cu13/lib'
    if [ -n "$old_runpath" ]; then
        new_runpath="$new_runpath:$old_runpath"
    fi
    patchelf --set-rpath "$new_runpath" python/mxnet/libmxnet.so
    echo "    new RUNPATH: $new_runpath"

    # Verify libmxnet declares a dependency on the CUDA driver (libcuda.so.1).
    # libmxnet references Driver-API symbols (e.g. cuLaunchKernel) directly, so
    # it MUST carry a DT_NEEDED for libcuda.so.1 — otherwise a clean wheel venv
    # aborts dlopen with "undefined symbol: cuLaunchKernel" (the dev tree only
    # works because the driver is already loaded globally).  This is supplied at
    # link time by `CUDA::cuda_driver` in CMakeLists.txt; the driver itself is
    # host-provided by the NVIDIA kernel driver and never bundled.  We do NOT use
    # `patchelf --add-needed` for this: it corrupts this ~1 GB binary (the .so
    # loads but segfaults during init).  Fail loudly if the NEEDED is missing so
    # the build is fixed at the source (link CUDA::cuda_driver) rather than
    # papered over.
    if readelf -d python/mxnet/libmxnet.so 2>/dev/null \
            | grep -qiE 'NEEDED.*\blibcuda\.so\.1\b'; then
        echo "==> libcuda.so.1 is a NEEDED dependency (from CUDA::cuda_driver link) — OK"
    else
        echo "  FAILED: libmxnet.so has no DT_NEEDED for libcuda.so.1." >&2
        echo "          Ensure CMakeLists.txt links CUDA::cuda_driver and rebuild." >&2
        exit 5
    fi
fi

OPENCV_DEPS_FLAG="${OPENCV_DEPS_FLAG:-1}"
if [ "$HAS_OPENCV" != 1 ]; then
    OPENCV_DEPS_FLAG=0
fi

# CUDA runtime pip dependencies are Linux-only; the macOS CPU wheel must not
# declare nvidia-*-cu13.  Cython is compiled into the Linux wheel for the _cy3
# fast path; on macOS we ship the ctypes path by default (set MXNET_WITH_CYTHON=1
# to opt in) to keep the wheel build robust across Xcode toolchains.
# ONNX (mxnet.onnx) is pure-Python and always ships in the wheel; the CUDA wheel
# AND the macOS CPU wheel additionally declare `onnx` as a hard runtime dependency
# (so a plain `pip install mxnet` has working ONNX export/import out of the box),
# while the Linux x86_64 CPU wheel keeps it as the optional `[onnx]` extra.  See
# setup.py _include_onnx_deps().
if [ "$FLAVOR" = macos ]; then
    CUDA_DEPS_FLAG=0
    ONNX_DEPS_FLAG="${MXNET_SETUP_ENABLE_ONNX_DEPS:-1}"
    CYTHON_FLAG="${MXNET_WITH_CYTHON:-0}"
elif [ "$FLAVOR" = linux-cpu ]; then
    # CPU wheel: no nvidia-*-cu13 deps; onnx stays the optional [onnx] extra
    # (the hard-onnx-dep policy is the CUDA + macOS wheels).  Cython is compiled in.
    CUDA_DEPS_FLAG=0
    ONNX_DEPS_FLAG="${MXNET_SETUP_ENABLE_ONNX_DEPS:-0}"
    CYTHON_FLAG="${MXNET_WITH_CYTHON:-1}"
else
    CUDA_DEPS_FLAG=1
    ONNX_DEPS_FLAG="${MXNET_SETUP_ENABLE_ONNX_DEPS:-1}"
    CYTHON_FLAG="${MXNET_WITH_CYTHON:-1}"
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
    MXNET_SETUP_ENABLE_OPENCV_DEPS="$OPENCV_DEPS_FLAG" \
    MXNET_SETUP_ENABLE_CUDA_DEPS="$CUDA_DEPS_FLAG" \
    MXNET_SETUP_ENABLE_ONNX_DEPS="$ONNX_DEPS_FLAG" \
    MXNET_WITH_CYTHON="$CYTHON_FLAG" \
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
# onnx is a hard runtime dependency on the wheel that requested it (the CUDA wheel);
# elsewhere mxnet.onnx still ships but onnx stays the optional [onnx] extra.
EXPECT_ONNX=off
[ "$ONNX_DEPS_FLAG" = 1 ] && EXPECT_ONNX=on
if [ "$FLAVOR" = macos ]; then
    # macOS CPU wheel: CUDA/cuDNN/NCCL off, oneDNN + OpenCV on.  Allow a dirty
    # tree because this script is typically run from a work-in-progress checkout
    # on the build host (the Linux release path runs from a clean, tagged tree).
    "$PYTHON_BIN" tools/release_provenance.py "$WHEEL" \
        --cmake-cache "$BUILD_DIR/CMakeCache.txt" \
        --package-version "$VERSION" \
        --expect-cuda off \
        --expect-cudnn off \
        --expect-nccl off \
        --expect-onednn on \
        --expect-opencv "$EXPECT_OPENCV" \
        --expect-onnx "$EXPECT_ONNX" \
        --expect-openmp on \
        --allow-dirty
elif [ "$FLAVOR" = linux-cpu ]; then
    # Linux x86_64 CPU wheel: CUDA/cuDNN/NCCL off, oneDNN + OpenCV on, onnx extra.
    "$PYTHON_BIN" tools/release_provenance.py "$WHEEL" \
        --cmake-cache "$BUILD_DIR/CMakeCache.txt" \
        --package-version "$VERSION" \
        --expect-cuda off \
        --expect-cudnn off \
        --expect-nccl off \
        --expect-onednn on \
        --expect-opencv "$EXPECT_OPENCV" \
        --expect-onnx "$EXPECT_ONNX" \
        --expect-openmp on
else
    "$PYTHON_BIN" tools/release_provenance.py "$WHEEL" \
        --cmake-cache "$BUILD_DIR/CMakeCache.txt" \
        --package-version "$VERSION" \
        --expect-cuda on \
        --expect-cudnn on \
        --expect-nccl on \
        --expect-onednn on \
        --expect-opencv "$EXPECT_OPENCV" \
        --expect-onnx "$EXPECT_ONNX" \
        --expect-openmp on
fi

echo "==> Wheel build OK: $WHEEL"
