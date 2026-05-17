# Building the Blackwell / CUDA 13 fork

This document describes the actual recipe used to produce the
`mxnet-2.0.0+cu13.bw.YYYYMMDD` release wheel. If you just want to *use*
MXNet on Blackwell, install the wheel from the GitHub release instead —
see [`README.md`](README.md).

The recipe targets a single GPU architecture (`sm_120`). Multi-arch
fatbin support is tracked in [`issues.md`](issues.md) item 31.

## Tested host

- Ubuntu 22.04 / 24.04, x86_64.
- AMD EPYC 7B12 (Zen 2, 64 threads) — no AVX-512, so bf16 falls back to
  fp32 emulation in oneDNN. Intel SPR or AMD Zen 4 / Granite Rapids will
  exercise the real bf16 path.
- NVIDIA RTX PRO 4000 / RTX 50-series (compute capability 12.0).
- NVIDIA driver R570 or newer.

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
| OpenCV      | 4.x        | optional, off in the release build             |

## Required apt packages

```bash
sudo apt update
sudo apt install -y \
  build-essential ninja-build cmake git \
  libopenblas-dev liblapack-dev \
  libnccl-dev libnccl2 \
  cuda-13 libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
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
  -DUSE_OPENCV=OFF \
  -DUSE_LAPACK=ON \
  -DUSE_BLAS=open \
  -DMXNET_CUDA_ARCH="12.0" \
  -DCMAKE_CUDA_ARCHITECTURES="120" \
  -DCMAKE_INSTALL_PREFIX=/opt/mxnet
```

Key flags:

- `MXNET_CUDA_ARCH="12.0"` and `CMAKE_CUDA_ARCHITECTURES="120"` together
  pin the build to Blackwell only. To add Ampere / Ada / Hopper, expand
  to `"8.0;8.6;8.9;9.0;12.0"` and `"80;86;89;90;120"` respectively. The
  release wheel intentionally stays single-arch to keep the artefact
  size manageable.
- `USE_F16C=ON` enables the F16C intrinsics path for fp16 (de)serialization.
- `USE_ONEDNN=ON` picks up the vendored v3.11 submodule.
- `USE_DIST_KVSTORE=OFF` skips ps-lite; not needed for single-host
  Blackwell development.

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

3. **`sm_120` PTX-only JIT stall.** The current build emits only
   `compute_120` PTX (no SASS). The first kernel launch per process
   pays a JIT compile penalty that can be hundreds of milliseconds.
   To pre-compile SASS, add
   `-gencode arch=compute_120,code=[compute_120,sm_120]` to the CUDA
   flags. Tracked in [`issues.md`](issues.md) item 20.

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

## Cross-references

- [`README.md`](README.md) — user-facing overview.
- [`CHANGELOG.md`](CHANGELOG.md) — per-release notes.
- [`issues.md`](issues.md) — open work list.
- Upstream Apache MXNet build docs under `docs/static_site/src/` are
  largely obsolete for this fork; treat them as historical reference.
