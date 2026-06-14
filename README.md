<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

MXNet 2.0 — CUDA 13 / Blackwell + Apple Silicon fork
====================================================

> Unofficial, maintained fork at [`smolix/mxnet`](https://github.com/smolix/mxnet).
> **Run existing MXNet code on current hardware** — NVIDIA Blackwell (and
> Ampere → Hopper) GPUs on the CUDA 13 stack, and native Apple Silicon CPU.

Apache MXNet was **archived on 2023-11-17**. The upstream tree is frozen at
CUDA 11 / cuDNN 8 / oneDNN v2 and does not build on Blackwell GPUs or modern
CUDA toolchains. This fork carries the patches needed to keep the residual MXNet
community — legacy research notebooks, frozen production pipelines, niche
operators like `_contrib_quantize_*`, and the [d2l.ai](https://d2l.ai) book —
running on today's hardware without a rewrite to PyTorch / JAX. It is **not** an
official Apache release.

- **Current release line:** `2.0.0+cu13.bw.<YYYYMMDD>` (latest published wheel
  `2.0.0+cu13.bw.20260614`) for Linux/CUDA, and `2.0.0+cpu.macos.<YYYYMMDD>` for
  Apple Silicon. The authoritative current build is always the newest on the
  [Releases page](https://github.com/smolix/mxnet/releases/latest).
- **What changed vs upstream:** see [`FIXED.md`](FIXED.md).
- **Known limitations / open work:** see [`OPEN_ISSUES.md`](OPEN_ISSUES.md).
- **Build from source:** see [`BUILDING.md`](BUILDING.md) and
  [`docs/cuda_wheel_build.md`](docs/cuda_wheel_build.md).

Quick install
-------------

**Linux + NVIDIA GPU (CUDA 13, driver R590+):**

```bash
# CPython 3.12 (a cp311 wheel is also published)
pip install "https://github.com/smolix/mxnet/releases/download/v2.0.0%2Bcu13.bw.20260614/mxnet-2.0.0%2Bcu13.bw.20260614-cp312-cp312-linux_x86_64.whl"
```

**macOS, Apple Silicon (CPU-only):**

```bash
# CPython 3.12
pip install "https://github.com/smolix/mxnet/releases/download/v2.0.0%2Bcu13.bw.20260614/mxnet-2.0.0%2Bcpu.macos.20260614-cp312-cp312-macosx_26_0_arm64.whl"
```

Smoke-test the install:

```bash
python -c "import mxnet as mx; from mxnet.runtime import Features; \
print(mx.__version__); print('OPENCV', Features().is_enabled('OPENCV')); \
print('GPUs', mx.device.num_gpus())"
```

See [Installation](#installation) for the runtime dependency model and
troubleshooting, and check the [Releases page](https://github.com/smolix/mxnet/releases/latest)
for a newer dated wheel before pinning a URL.

What works
----------

* **Blackwell `sm_120` + full Ampere→Blackwell fatbin** (CUDA 13.0): `sm_80`
  (A100), `sm_86` (RTX 30xx), `sm_89` (Ada/RTX 40xx), `sm_90` (Hopper),
  `sm_100` (B100/B200), `sm_120` (RTX 50xx) SASS + `compute_120` PTX fallback.
* **cuDNN 9.x** including the rewritten v8-style RNN path (LSTM/GRU/RNN, fwd+bwd).
  TF32 on by default for FP32 conv (PyTorch/TF default; ~2.87× on `sm_120`).
* **NCCL 2.28** single-process / multi-GPU.
* **oneDNN v3.11** float backend everywhere; full INT8 path on x86 (per-OC weight
  scales, fused conv/FC, fused sum, dequant-to-fp32).
* **cuBLASLt GEMM** (fp32/fp16/fp64), bitwise-parity-verified vs the legacy path.
* **CUDA Graphs** — revived and **default-on for hybridized cached-ops** with
  `static_alloc=True` + `static_shape=True` (eager execution unchanged); measured
  1.5–2.3× on transformer/RNN-ish nets, bitwise-identical results.
* **INT8 quantization** (`quantize_net`, `_sg_onednn_conv`,
  `_sg_onednn_fully_connected`) on x86; fp16/fp32 forward + backward training.
* **AMP (automatic mixed precision) subgraph** — on CPUs without AVX-512-BF16 the
  bf16 subgraph ops fall back to fp32 (all 6 AMP subgraph tests pass).
* **ONNX export/import** (opset-13 default, ONNX 1.21 / ORT 1.24) — *in source
  builds*; the published wheels are built ONNX-free (see [`OPEN_ISSUES.md`](OPEN_ISSUES.md)).
* **Self-contained wheels.** Both the Linux CUDA wheel and the macOS CPU wheel
  bundle OpenCV and its **full transitive closure** into `mxnet/lib/` (ELF
  `$ORIGIN` on Linux, Mach-O `@loader_path` on macOS), so `import mxnet` reports
  `OPENCV=True` and `mx.image` native decode/resize work on a clean host with no
  system OpenCV.
* **Native macOS arm64 CPU wheel** — Accelerate BLAS/LAPACK + float oneDNN.
  ~14.9k unittest / operator / NumPy / Gluon / quantization-API tests pass
  (`tools/run_macos_wheel_full_test.sh`).

What is experimental or not covered
-----------------------------------

The five most likely to affect you (full list and details in
[`OPEN_ISSUES.md`](OPEN_ISSUES.md)):

1. **CUDA 13.0 / driver R580 is unsupported** — the wheel pins
   `nvidia-cublas>=13.5`, which needs **driver R590+**. On R580 large GEMMs fail
   with `CUBLAS_STATUS_NOT_INITIALIZED`.
2. **ONNX is not in the published wheels** (fixed in source only).
3. **Apple Silicon: oneDNN INT8 + subgraph fusion are gated off** (the
   Xbyak_aarch64 JIT is unreliable on Apple Silicon — see
   `SupportDNNLAArch64JITPrimitives` in `src/operator/nn/dnnl/dnnl_base-inl.h`);
   those ops fall back to native kernels and the `tests/python/dnnl` fusion/quant
   lane does not apply.
4. **bf16 on CPUs without AVX-512-BF16** is emulated in fp32 — correct, not faster.
5. **Backward through quantized ops is unvalidated** — forward INT8 inference is
   solid; quantized *training* is not verified.

System requirements
-------------------

**Linux / CUDA:**

* Linux x86_64 (tested on Ubuntu 22.04 / 24.04).
* **NVIDIA driver R590 or newer** (required by the `nvidia-cublas>=13.5` pin; the
  older CUDA 13.0 / R580 line is not supported — see
  [`OPEN_ISSUES.md`](OPEN_ISSUES_DETAILS.md#oi-19)).
* CUDA 13.0 toolkit at `/usr/local/cuda/` — supplies the base runtime libs
  (`libcudart`, `libcublas`, `libcufft`, `libcusolver`, `libcurand`, `libnvrtc`)
  that NVIDIA does not yet ship as real `cu13` PyPI wheels.
* cuDNN and NCCL are pulled in automatically as pip deps
  (`nvidia-cudnn-cu13>=9.22,<10`, `nvidia-nccl-cu13>=2.28,<3`,
  `nvidia-cublas>=13.5,<14`).
* CPython **3.11 or 3.12** (both published as wheels); 3.10–3.13 for source builds.

**macOS:**

* Apple Silicon (arm64), macOS 13+; CPython 3.12 (published wheel). CPU-only.

Installation
------------

The wheels are not 2 GB monoliths — only OpenCV is bundled. Everything else is a
declared pip dependency or comes from your system CUDA toolkit:

* **CUDA / cuDNN / NCCL** — `pip` pulls `nvidia-cudnn-cu13` (~1 GB) and
  `nvidia-nccl-cu13` (~190 MB) from PyPI (the PyTorch/JAX layout). The base CUDA
  13 runtime libs come from your system toolkit at `/usr/local/cuda/`
  (`apt install cuda-13`).
* **OpenBLAS** — from the `scipy-openblas32` wheel.
* **OpenCV** — bundled *inside* the wheel (`mxnet/lib/`), reached via `$ORIGIN`
  (Linux) / `@loader_path` (macOS). You do **not** need `libopencv-dev` or to
  `pip install opencv-python` for the native C++ image path; the macOS wheel does
  additionally depend on `opencv-python` for the Python `cv2` helpers.

The Linux CUDA wheel is ~454 MB; a smaller `BUNDLE_OPENCV=0` / `USE_OPENCV=OFF`
build is possible if you do not need MXNet's native image path (reports
`OPENCV=False`).

**Troubleshooting:**

| Symptom | Cause / fix |
|---|---|
| `CUBLAS_STATUS_NOT_INITIALIZED` on a non-trivial `dot`/`FullyConnected` | driver too old; upgrade to **R590+** ([OI-19](OPEN_ISSUES_DETAILS.md#oi-19)) |
| `cuDNN lib mismatch: …` printed on first GPU use | harmless minor-version note (wheel built vs 9.23, pin resolves 9.22) ([OI-20](OPEN_ISSUES_DETAILS.md#oi-20)) |
| `cudaErrorNoKernelImageForDevice` / `no kernel image available` | wheel lacks SASS for your GPU — rebuild with your arch in `MXNET_CUDA_ARCH` |
| `Build with USE_OPENCV=1 for image io` | OpenCV-off wheel — install/build an `OPENCV=ON` wheel |
| slow batch-size-1 CPU inference | set `OMP_NUM_THREADS=1` for bs=1 ([OI-14](OPEN_ISSUES_DETAILS.md#oi-14)) |

Building from source
--------------------

See [`BUILDING.md`](BUILDING.md) for the from-scratch recipe (Linux/CUDA and the
Apple Silicon CPU build) and [`docs/cuda_wheel_build.md`](docs/cuda_wheel_build.md)
for the authoritative, provenance-gated release-wheel pipeline. The short version
for a CUDA build: clone with submodules, install `libnccl-dev` and
`libcudnn9-dev-cuda-13` *before* invoking `cmake`, then

```bash
cmake -S . -B build -G Ninja -DUSE_CUDA=ON \
  -DMXNET_CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;12.0+PTX" ...
```

Acknowledgements
----------------

This fork builds on the work of the Apache MXNet community and its contributors.
All upstream code is Apache 2.0; the CUDA 13 / Blackwell / Apple Silicon patches
in this fork are likewise Apache 2.0.

---

<details>
<summary><b>Upstream Apache MXNet (archived 2023-11-17) — historical</b></summary>

Apache MXNet is a deep learning framework designed for both *efficiency* and
*flexibility*. It lets you mix
[symbolic and imperative programming](https://mxnet.apache.org/api/architecture/program_model),
features a dynamic dependency scheduler that parallelizes operations on the fly,
a NumPy-like interface integrated with the Gluon 2.0 API, and automatic
hybridization. It scaled to multiple GPUs and machines via
[ps-lite](https://github.com/dmlc/ps-lite), [Horovod](https://github.com/horovod/horovod),
and [BytePS](https://github.com/bytedance/byteps), with bindings for Python,
Java, C++, R, Scala, Clojure, Go, Javascript, Perl, and Julia.

The upstream project was **archived on 2023-11-17**. Its website
([mxnet.apache.org](https://mxnet.apache.org/)), CI dashboards, mailing lists,
Slack, and social channels are no longer actively monitored; links to them are
kept only for historical reference.

MXNet emerged from a collaboration by the authors of
[cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva),
and [purine2](https://github.com/purine/purine2).

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015.

Licensed under [Apache-2.0](LICENSE).

</details>
