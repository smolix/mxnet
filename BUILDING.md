# Building the Blackwell / CUDA 13 fork

This document describes the actual recipe used to produce the
`mxnet-2.0.0+cu13.bw.YYYYMMDD` release wheel. If you just want to *use*
MXNet on Blackwell, install the wheel from the GitHub release instead —
see [`README.md`](README.md).

> **The authoritative, end-to-end release-wheel recipe now lives in
> [`docs/cuda_wheel_build.md`](docs/cuda_wheel_build.md)** — it wraps the
> whole build → bundle OpenCV → package → *verify* pipeline behind
> `tools/build_cleanup_wheel.sh` and a provenance gate. Prefer it for
> producing a release wheel. This file is kept for the manual/legacy
> recipe and the macOS arm64 CPU smoke build.

The recipe targets an explicit multi-arch CUDA fatbin: `sm_80`, `sm_86`,
`sm_89`, `sm_90`, `sm_100`, and `sm_120` SASS, plus `compute_120` PTX
fallback. This is the release-wheel matrix used to keep Ampere, Ada,
Hopper, and Blackwell (both datacenter `sm_100` and consumer `sm_120`)
coverage explicit. **OpenCV is built `ON`** and bundled into the wheel —
the C++ image I/O path (`mx.image`, `gluon.data.vision`) requires it and
there is no Python-level fallback; see `docs/cuda_wheel_build.md` §1.

## Tested host

- Ubuntu 22.04 / 24.04, x86_64.
- AMD EPYC 7B12 (Zen 2, 64 threads) — no AVX-512, so bf16 falls back to
  fp32 emulation in oneDNN. Intel SPR or AMD Zen 4 / Granite Rapids will
  exercise the real bf16 path.
- NVIDIA RTX PRO 4000 / RTX 50-series (compute capability 12.0).
- NVIDIA driver R570 or newer.
- macOS arm64 is covered by the CPU-only smoke path with oneDNN enabled. It is
  not a CUDA release-wheel target.

A full clean build takes roughly **35-50 minutes** on 64 threads. The
CUDA compile phase dominates; expect `nvcc` to be the long pole.

## Toolchain

| Component   | Version    | Notes                                          |
| ----------- | ---------- | ---------------------------------------------- |
| CUDA        | 13.0       | system install at `/usr/local/cuda-13`         |
| cuDNN       | 9.22.0     | local `cudnn_local/unpacked/nvidia/cudnn/` from `nvidia-cudnn-cu13==9.22.0.52` wheel (system 9.14 still works too) |
| NCCL        | 2.28.3     | `libnccl2` + **`libnccl-dev`** (see gotcha 1)  |
| oneDNN      | 3.11       | vendored as submodule under `3rdparty/onednn`  |
| GCC         | 11 - 13    | 12 used for the release wheel                  |
| CMake       | 3.27+      | older 3.16 in CMakeLists.txt is too lax        |
| Python      | 3.10-3.13  | 3.11 used for the release wheel                |
| OpenBLAS    | 0.3.x      | `libopenblas-dev`                              |
| OpenCV      | 4.6        | **ON** in the release build (`libopencv-dev`); native libs bundled into the wheel — required for image I/O |
| patchelf    | any        | RUNPATH patching + OpenCV bundling             |

## Required apt packages

```bash
sudo apt update
sudo apt install -y \
  build-essential ninja-build cmake git patchelf \
  libopenblas-dev liblapack-dev \
  libnccl-dev libnccl2 \
  cuda-13 libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
  libopencv-dev \
  python3-dev python3-pip
```

`libnccl-dev` is the one most likely to bite — see gotcha 1 below.

## Clone with submodules

```bash
git clone --recursive git@github.com:smolix/mxnet.git
cd mxnet
git submodule update --init --recursive
```

Without `--recursive` you will get a confusing failure when oneDNN
headers are missing.

## CMake configure

```bash
mkdir build && cd build

cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_CUDNN=ON \
  -DUSE_NCCL=ON \
  -DUSE_DIST_KVSTORE=OFF \
  -DUSE_ONEDNN=ON \
  -DUSE_OPENMP=ON \
  -DUSE_F16C=ON \
  -DUSE_OPENCV=ON \
  -DUSE_LAPACK=ON \
  -DUSE_BLAS=open \
  -DMXNET_CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;12.0+PTX" \
  -DCMAKE_INSTALL_PREFIX=/opt/mxnet
```

Key flags:

- `MXNET_CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;12.0+PTX"` is the release matrix:
  `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, and `sm_120` SASS, plus
  `compute_120` PTX fallback. Note that Blackwell is **two** non-compatible
  families — datacenter `sm_100` and consumer `sm_120` — and `compute_120`
  PTX does *not* JIT down to `sm_100`, so both must be listed explicitly to
  cover B200 *and* RTX 50-series. Leave `CMAKE_CUDA_ARCHITECTURES` unset;
  the top-level CMake config sets it to `OFF` so MXNet's
  `CUDA_SELECT_NVCC_ARCH_FLAGS` emits the fatbin matrix from
  `MXNET_CUDA_ARCH`.
- `USE_OPENCV=ON` is **required** for the release wheel: `mx.image` and
  `gluon.data.vision` decode/resize images through OpenCV at the C++ layer.
  An OpenCV-off wheel raises `Build with USE_OPENCV=1 for image io` at
  runtime with no Python fallback. The native `libopencv_*.so` files are
  bundled into the wheel by `tools/build_cleanup_wheel.sh`; see
  [`docs/cuda_wheel_build.md`](docs/cuda_wheel_build.md).
- `USE_F16C=ON` enables the F16C intrinsics path for fp16 (de)serialization.
- `USE_ONEDNN=ON` picks up the vendored v3.11 submodule.
- `USE_DIST_KVSTORE=OFF` skips ps-lite; not needed for single-host
  Blackwell development.

## Apple Silicon CPU-only smoke build

For native macOS arm64 validation, use a separate build directory and the
minimal CPU-only feature set. This avoids the Linux/CUDA release settings and
does not use the shared build scripts.

OpenMP is recommended for CPU performance (it also switches oneDNN from the
single-threaded `SEQ` runtime to multi-threaded `OMP`). AppleClang ships no
OpenMP runtime, so build the hermetic libomp once; it installs under `.deps/`
and is auto-discovered on the next configure (no `-DOPENMP_ROOT` needed):

```bash
python tools/dependencies/build_openmp.py
```

```bash
cmake -S . -B build-macos-arm64 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DUSE_CUDA=OFF \
  -DUSE_CUDNN=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_ONEDNN=ON \
  -DUSE_OPENMP=ON \
  -DUSE_OPENCV=OFF \
  -DUSE_BLAS=apple \
  -DUSE_LAPACK=ON \
  -DUSE_DIST_KVSTORE=OFF \
  -DUSE_SSE=OFF \
  -DUSE_F16C=OFF \
  -DBUILD_CPP_EXAMPLES=OFF

cmake --build build-macos-arm64 --target mxnet -- -j 3
export MXNET_LIBRARY_PATH="$(pwd)/build-macos-arm64/libmxnet.dylib"
uv venv .venv --python 3.11
# scipy is required by the broader tests/python/unittest suite (sparse, random,
# metric, image, numpy_op, gluon probability, ...) — those modules import it at
# collection time, so it must be present or ~8 files error out. The curated
# apple_silicon_cpu_smoke subset itself does not need scipy. "numpy<2" keeps the
# resolver on a scipy build compatible with the pinned NumPy.
uv pip install --python .venv/bin/python "numpy<2" scipy requests pytest pytest-timeout
MXNET_SETUP_ENABLE_CUDA_DEPS=0 uv pip install --python .venv/bin/python -e ./python
```

### OpenMP under a fully UV-managed toolchain

The smoke recipe above already enables OpenMP. If you also want CMake/Ninja
themselves isolated under the repo (no system cmake), run the same steps through
UV. `build_openmp.py` installs `libomp` into a repo-local `.deps/` prefix that is
auto-discovered on configure, so the `-DOPENMP_ROOT` below is an explicit
override and can be omitted:

```bash
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
  uv run --python .venv/bin/python --with cmake --with ninja \
    python tools/dependencies/build_openmp.py

UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
  uv run --python .venv/bin/python --with cmake --with ninja \
    cmake -S . -B build-macos-arm64-openmp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DUSE_CUDA=OFF \
  -DUSE_CUDNN=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_ONEDNN=ON \
  -DUSE_OPENMP=ON \
  -DOPENMP_ROOT="$(pwd)/.deps/openmp-22.1.5-macos-arm64" \
  -DUSE_OPENCV=OFF \
  -DUSE_BLAS=apple \
  -DUSE_LAPACK=ON \
  -DUSE_DIST_KVSTORE=OFF \
  -DUSE_SSE=OFF \
  -DUSE_F16C=OFF \
  -DBUILD_CPP_EXAMPLES=OFF
```

### Optional OpenCV via UV

The arm64 macOS smoke recipe keeps OpenCV off by default, but the image and
vision tests can be enabled without Homebrew, MacPorts, or system OpenCV by
building a repo-local OpenCV through UV. The same helper also supports Linux
and installs under a platform/architecture-specific `.deps/` prefix.

```bash
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
  uv run --python .venv/bin/python --with cmake --with ninja \
    python tools/dependencies/build_libturbojpeg.py

UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
  uv run --python .venv/bin/python --with cmake --with ninja \
    python tools/dependencies/build_opencv.py

UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
  uv run --python .venv/bin/python --with cmake --with ninja \
    cmake -S . -B build-macos-arm64-opencv -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DUSE_CUDA=OFF \
  -DUSE_CUDNN=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_ONEDNN=ON \
  -DUSE_OPENMP=OFF \
  -DUSE_OPENCV=ON \
  -DOPENCV_ROOT="$(pwd)/.deps/opencv-4.9.0-macos-arm64" \
  -DOpenCV_DIR="$(pwd)/.deps/opencv-4.9.0-macos-arm64/lib/cmake/opencv4" \
  -DUSE_LIBJPEG_TURBO=ON \
  -DTURBOJPEG_ROOT="$(pwd)/.deps/libjpeg-turbo-3.0.4-macos-arm64" \
  -DUSE_BLAS=apple \
  -DUSE_LAPACK=ON \
  -DUSE_DIST_KVSTORE=OFF \
  -DUSE_SSE=OFF \
  -DUSE_F16C=OFF \
  -DBUILD_CPP_EXAMPLES=OFF \
  -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python"

UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python \
  uv run --python .venv/bin/python --with cmake --with ninja \
    cmake --build build-macos-arm64-opencv --target mxnet im2rec -- -j 3
export MXNET_LIBRARY_PATH="$(pwd)/build-macos-arm64-opencv/libmxnet.dylib"
```

The helpers build libjpeg-turbo 3.0.4 and OpenCV 4.9.0 under `.deps/`. OpenCV
is configured with bundled image codec dependencies; MXNet also links directly
against libjpeg-turbo for the JPEG RecordIO fast path. On macOS OpenCV uses
Apple SDK zlib to avoid SDK conflicts; on Linux it builds zlib with OpenCV. It
ignores `/opt/local` and `/usr/local` during OpenCV configuration so MacPorts,
Homebrew, and ad hoc local installs do not bleed into the dependency tree.
MXNet CMake is pointed at the resulting prefixes via `OPENCV_ROOT` and
`TURBOJPEG_ROOT`.

If the checkout path contains shell-special characters such as spaces or
parentheses, the helper re-enters through a stable `/private/tmp/mxnet-opencv-*`
symlink before invoking OpenCV's CMake build. The installed files still live
under the checkout's `.deps/` directory.

Run the smoke subset tracked in test metadata:

```bash
grep -Ev '^\s*(#|$)' tests/python/apple_silicon_cpu_smoke \
  | xargs .venv/bin/python -m pytest -v --timeout=180 --tb=short
```

The list currently covers base, engine, NumPy smoke, Gluon smoke, and a minimal
oneDNN execution test.

## Build

```bash
ninja -j $(nproc)
```

On 64 threads expect 35-50 minutes. If you only need to iterate on a
single CUDA file, `ninja src/.../foo.cu.o` is much faster than a full
rebuild.

## Python wheel

```bash
cd ../python
pip install -e .
# or, for a release artefact:
python setup.py bdist_wheel
```

The wheel will be tagged with the version string from
`python/mxnet/libinfo.py` — currently `2.0.0+cu13.bw.20260517`. Update
that file (and re-run `bdist_wheel`) whenever you cut a new dated
release.

## Verification

```bash
python -c "
import mxnet as mx
print('version', mx.__version__)
print('compute capability', mx.runtime.feature_list())
x = mx.nd.ones((3, 3), ctx=mx.gpu())
print(x.asnumpy())
"
```

Expected output: version string ends in `+cu13.bw.YYYYMMDD`, a 3x3 matrix
of ones, no `CUDNN_STATUS_ARCH_MISMATCH` / `no kernel image available`
errors.

For a deeper smoke run, execute one of the DNNL subgraph test files:

```bash
cd ../tests/python/dnnl/subgraphs
pytest test_conv_subgraph.py -x -v
```

A clean run reports roughly 815 pass / 3 skip on this build. Anything
that errors out at collect time (e.g. ONNX, `test_amp_subgraph`) is
expected — see [`issues.md`](issues.md).

## Gotchas

1. **Install `libnccl-dev` BEFORE running `cmake`.** Without the headers,
   the CMake NCCL probe silently disables `USE_NCCL` even when
   `-DUSE_NCCL=ON` is passed. The resulting wheel will throw
   `MXNetError: NCCL is disabled` at the first multi-GPU `kvstore`
   call. Rerunning cmake from a clean `build/` directory is the fix.

2. **CUDA 13 + GCC version pinning.** CUDA 13 supports GCC 12 and 13.
   If your distro defaults to GCC 14, set `CXX=g++-13 CC=gcc-13` before
   cmake or NVCC will reject host headers.

3. **Avoid `sm_120` PTX-only builds.** The release matrix entry
   `12.0+PTX` emits both `sm_120` SASS and `compute_120` PTX. A
   PTX-only Blackwell build pays a first-launch JIT compile penalty that
   can be hundreds of milliseconds per process.

4. **bf16 on AMD Zen 2 / older Intel.** oneDNN v3 still supports bf16
   primitives, but on CPUs without AVX-512-BF16 it silently emulates
   them in fp32 — so bf16 numerics are *correct* but the perf is no
   better than fp32. Not a build error.

5. **cuDNN heuristic gap (now narrow).** cuDNN 9.0 - 9.20 ship `sm_120`
   heuristic tables with incomplete coverage. The 2026-05-17 release
   wheel ships 9.22 which closes the depthwise gap (depthwise 3×3
   256→256: 0.16 → 1.14 TFLOPS, ~7×); other shapes are within noise of
   9.14. The build is happy with 9.0+ but cuDNN 9.22 is recommended.

6. **`libnccl2` and `libcudnn9-cuda-13` ABI lock.** The wheel binds to
   the exact `SONAME` of these libraries at link time. If you upgrade
   cuDNN to 9.15+ later, the existing wheel still works (cuDNN keeps
   `libcudnn.so.9`). A major-version bump (cuDNN 10, NCCL 3) will
   require a rebuild.

7. **`ccache` is your friend.** A second clean build with `ccache`
   warmed drops to roughly 10-15 minutes. `CXX="ccache g++" CC="ccache
   gcc"` before cmake is enough.

## Third-party / submodule warning policy (CN9)

This fork carries vendored copies of `dmlc-core`, `onednn`, and `tvm` as
submodules under `3rdparty/`.  Two specific build-time warnings come from
inside those submodules and **are not patched in this repository**:

- **Bundled dmlc concurrent queue** (`3rdparty/dmlc-core/include/dmlc/...`)
  assigns `-1` into a `uint32_t` sentinel.  NVCC emits an
  unsigned-conversion warning.  The behavior is intentional in dmlc; we
  do not maintain a private dmlc-core fork.
- **oneDNN vendored ITT assembly** (`3rdparty/onednn/.../ittptmark64.S.o`)
  is built without a `.note.GNU-stack` section, so the linker emits an
  executable-stack warning.  oneDNN owns the upstream fix; carrying a
  private patch in our submodule pointer would dirty the detached tree
  with no upstream PR to converge on.

Both warnings are documented as **CN9** in `issues.md` (Resolved /
informational) and are not blockers.  If you want to silence them
locally:

- Update the submodule pointer to a newer oneDNN/dmlc commit if upstream
  fixes them later.
- Apply the patch in your own working tree but **do not commit** it to
  the fork — submodule pointer changes here imply an upstream
  responsibility this project doesn't accept.

If a future release of oneDNN or dmlc-core lands the upstream fix, the
fork's next submodule bump will pick it up automatically.

## Cross-references

- [`README.md`](README.md) — user-facing overview.
- [`CHANGELOG.md`](CHANGELOG.md) — per-release notes.
- [`issues.md`](issues.md) — open work list.
- Upstream Apache MXNet build docs under `docs/static_site/src/` are
  largely obsolete for this fork; treat them as historical reference.
