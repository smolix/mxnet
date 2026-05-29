<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Building the CUDA release wheel (`mxnet-2.0.0+cu13.bw.*`)

This is the authoritative, end-to-end recipe for producing the
**Ampere-through-Blackwell CUDA 13 release wheel** that the
[d2l.ai](https://d2l.ai) book and other downstreams consume. It is meant
to be followed top to bottom by a human or an agent and to leave you with
a wheel that is *verified correct* before it is published — not merely one
that imports.

> **If you only want to *use* MXNet**, do not build anything. Install the
> published wheel from the GitHub release (see [`README.md`](../README.md)).
> This document is for *producing* that wheel.

For the legacy/manual recipe and the macOS arm64 CPU smoke build see
[`BUILDING.md`](../BUILDING.md); that file now defers to this one for the
Linux/CUDA release wheel.

---

## 0. TL;DR — the one command that matters

Once the host has the prerequisites (§2) and the build directory has been
configured once (§3), the entire build → bundle → package → verify
pipeline is a single script:

```bash
# from the repo root, with .venv-mxnet on PATH or as $PYTHON
MXNET_BUILD_JOBS=64 tools/build_cleanup_wheel.sh 2.0.0+cu13.bw.$(date -u +%Y%m%d).1
```

That script (§4) refreshes CMake, compiles `libmxnet.so`, **bundles the
system OpenCV libraries into the wheel**, patches the RUNPATH, builds the
wheel, and finally runs `tools/release_provenance.py` which **fails the
build** unless the wheel actually has the feature set we promise
(CUDA + cuDNN + NCCL + oneDNN + **OpenCV**, all *on*).

If `build_cleanup_wheel.sh` exits 0, the wheel in `dist/` is releasable.
If it exits non-zero, **do not publish** — read the error; the provenance
gate exists specifically to stop a half-featured wheel (see §7) from ever
reaching a user.

---

## 1. Why this document exists: the OpenCV regression

A release wheel once shipped built with `USE_OPENCV=OFF`. It imported
fine, passed a smoke test, and was published. Then **27 notebooks in the
d2l book failed** at runtime with:

```
MXNetError: Build with USE_OPENCV=1 for image io.
MXNetError: Build with USE_OPENCV=1 for image resize operator.
```

That string is the `#else` branch inside `libmxnet.so`: it fires when the
binary was compiled *without* OpenCV. `mx.image.imdecode`,
`mx.image.imresize`, and every `gluon.data.vision` transform that decodes
or resizes an image route through OpenCV at the C++ layer. With OpenCV
compiled out, there is no Python-level workaround — the capability is
simply absent from the binary.

The lesson, now enforced in tooling:

1. **The CUDA release wheel must be built with `USE_OPENCV=ON`.** It is
   not optional for downstreams that touch images (i.e. all of computer
   vision).
2. **OpenCV must be self-contained in the wheel.** The native
   `libopencv_*.so` files are *bundled* into `mxnet/lib/` and reached via
   RUNPATH — we do **not** rely on the user pip-installing `opencv-python`
   to supply them (see §6).
3. **The build cannot be trusted to be correct by inspection.** The
   `release_provenance.py` gate (§7) re-derives the feature set from the
   compiled binary and the wheel contents and refuses anything that
   doesn't match. A future OpenCV-off regression is caught here, at build
   time, instead of in a user's notebook.

There is also a unit test, `test_d2l_opencv_image_io_regression.py`, that
round-trips a PNG through `mx.image` and `gluon` so the capability is
exercised in CI.

---

## 2. Prerequisites (host)

The reference host is Ubuntu 24.04, x86_64, with NVIDIA GPUs and a CUDA 13
toolkit. The wheel is a *fat* binary (multi-arch SASS, see §3) so the GPU
you build *on* does not have to match the GPUs you build *for* — e.g. you
can build the full Blackwell-capable wheel on an Ada (RTX 4090) box and it
will still carry `sm_100`/`sm_120` SASS for machines you don't have.

| Component | Version (reference) | Where it comes from |
|-----------|--------------------|---------------------|
| CUDA toolkit | 13.0 | `/usr/local/cuda-13` |
| cuDNN | 9.22 | system `libcudnn9-dev-cuda-13` *or* a local wheel unpack |
| NCCL | 2.28 | system `libnccl-dev` + `libnccl2` |
| oneDNN | vendored | `3rdparty/onednn` submodule |
| OpenCV | 4.6 | system `libopencv-dev` (core + imgproc + imgcodecs) |
| OpenBLAS | 0.3.x | system `libopenblas-dev` |
| CMake | ≥ 3.27 | apt |
| Ninja | any | apt |
| patchelf | any | apt |
| GCC | 11–13 | apt |
| Python | 3.12 | the `.venv-mxnet` used by the consumer |

```bash
sudo apt update
sudo apt install -y \
  build-essential ninja-build cmake git patchelf \
  libopenblas-dev liblapack-dev \
  libnccl-dev libnccl2 \
  libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
  libopencv-dev \
  python3-dev python3-pip
```

`libnccl-dev` (not just `libnccl2`) and `libcudnn9-dev-cuda-13` (the
`-dev` headers) are the two most commonly missing pieces — without them
CMake silently builds *without* the feature and the provenance gate will
(correctly) reject the wheel.

### Submodules

A fresh clone **must** initialise submodules or CMake fails at
`3rdparty/googletest` / `3rdparty/onednn`:

```bash
git submodule update --init --recursive
```

### Python build/test environment

The build script defaults `$PYTHON` to `<repo>/.venv-mxnet/bin/python` if
present, else `python3`. That interpreter needs the PEP 517 build
front-end and `wheel`:

```bash
uv pip install --python .venv-mxnet/bin/python build wheel
```

The native library is compiled by CMake/Ninja, **not** by `setup.py`;
`python -m build` only packages the already-built `libmxnet.so` plus the
bundled libs. So the Python env needs nothing heavyweight to *build* —
`numpy<2`, `requests`, `graphviz`, `packaging`, and `scipy-openblas32`
(all already present in `.venv-mxnet`) are the *runtime* deps declared in
`setup.py`.

---

## 3. The architecture matrix (Ampere → Blackwell)

The CUDA fatbin targets are set by `MXNET_CUDA_ARCH`. The release value is:

```
MXNET_CUDA_ARCH = "8.0;8.6;8.9;9.0;10.0;12.0+PTX"
```

| Code | SASS | GPUs |
|------|------|------|
| `8.0` | `sm_80` | Ampere datacenter — A100 |
| `8.6` | `sm_86` | Ampere consumer — RTX 30xx, A40 |
| `8.9` | `sm_89` | Ada — RTX 40xx, L40/L40S |
| `9.0` | `sm_90` | Hopper — H100/H200 |
| `10.0` | `sm_100` | Blackwell datacenter — B100/B200/GB200 |
| `12.0` | `sm_120` | Blackwell consumer — RTX 50xx, RTX PRO 6000 |
| `12.0+PTX` | `compute_120` PTX | forward-compat JIT for GPUs newer than `sm_120` |

Notes on why the list looks the way it does:

- **Blackwell is two non-compatible families.** `sm_100` (datacenter) and
  `sm_120` (consumer) are *not* binary-compatible with each other, and
  `compute_120` PTX does **not** JIT down to `sm_100`. To cover real B200
  *and* RTX 50-series hardware you need **both** `10.0` and `12.0` SASS
  explicitly. Dropping `10.0` (as an earlier matrix did) silently loses
  datacenter Blackwell.
- **PTX only on the top arch.** A single `compute_120` PTX entry gives
  forward compatibility (the driver JITs it) for *future* archs ≥ `sm_120`
  without bloating the fatbin with PTX for every level.
- **`CMAKE_CUDA_ARCHITECTURES` is left `OFF`.** The top-level CMake sets it
  to `OFF` on purpose so that MXNet's own `CUDA_SELECT_NVCC_ARCH_FLAGS`
  (`cmake/upstream/select_compute_arch.cmake`) emits the `-gencode` matrix
  from `MXNET_CUDA_ARCH`. Don't set `CMAKE_CUDA_ARCHITECTURES` by hand.

You can confirm the emitted flags in the configure log:

```
-- CUDA: Using the following NVCC architecture flags
   -gencode;arch=compute_80,code=sm_80; ... ;
   -gencode;arch=compute_100,code=sm_100;
   -gencode;arch=compute_120,code=sm_120;
   -gencode;arch=compute_120,code=compute_120
```

CUDA 12.8+ is required for `sm_100`/`sm_120` to be known to nvcc; CUDA 13.0
satisfies this.

### First-time configure

`build_cleanup_wheel.sh` *re-uses* an existing `build/` and only refreshes
the feature toggles; it does **not** set the generator, build type, BLAS
vendor, or arch. So the very first configure of a clean tree must set
those:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_NCCL=ON \
  -DUSE_ONEDNN=ON -DUSE_OPENMP=ON -DUSE_F16C=ON \
  -DUSE_OPENCV=ON \
  -DUSE_LAPACK=ON -DUSE_BLAS=open \
  -DUSE_DIST_KVSTORE=OFF \
  -DMXNET_CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;12.0+PTX"
```

`MXNET_CUDA_ARCH` and `USE_OPENCV` are then sticky in `build/CMakeCache.txt`
and survive the script's reconfigure. Verify:

```bash
grep -E 'USE_OPENCV:BOOL|MXNET_CUDA_ARCH' build/CMakeCache.txt
# USE_OPENCV:BOOL=ON
# MXNET_CUDA_ARCH:STRING=8.0;8.6;8.9;9.0;10.0;12.0+PTX
```

A full clean compile is ~1300 Ninja steps; the six-arch CUDA fatbin
compile dominates. Budget **45–70 min on 64 threads**; the GPU `.cu` files
are the long pole. Use `MXNET_BUILD_JOBS=64` (or your core count) to
saturate the box.

---

## 4. What `build_cleanup_wheel.sh` does

`tools/build_cleanup_wheel.sh [<version>]` is the single source of truth
for the release build. In order:

1. **Refresh CMake** with the release feature toggles
   (`USE_CUDA/CUDNN/NCCL/ONEDNN/OPENCV=ON`). Arch + build type + BLAS come
   from the cached first-time configure (§3).
2. **Compile** `--target mxnet --parallel $MXNET_BUILD_JOBS`.
3. **Probe the cache** for `USE_OPENCV:BOOL=ON` → `HAS_OPENCV`.
4. **Stage** `build/libmxnet.so` into `python/mxnet/`.
5. **Bundle OpenCV** (when `HAS_OPENCV` and `BUNDLE_OPENCV=1`, the default):
   resolve the `libopencv_*` SONAMEs that `libmxnet.so` actually `NEEDED`,
   copy the real system `.so` files into `python/mxnet/lib/`, and recreate
   the SONAME symlinks (e.g. `libopencv_imgcodecs.so.406`). Transitive
   `libopencv_*` deps are followed too.
6. **Patch RUNPATH** on `libmxnet.so` to:
   ```
   $ORIGIN/lib                       # bundled OpenCV
   $ORIGIN/../scipy_openblas32/lib   # OpenBLAS from the scipy-openblas32 wheel
   $ORIGIN/../nvidia/cudnn/lib       # cuDNN from nvidia-cudnn-cu13
   $ORIGIN/../nvidia/nccl/lib        # NCCL from nvidia-nccl-cu13
   $ORIGIN/../nvidia/cu13/lib        # CUDA runtime, if present as a wheel
   ```
7. **Build the wheel** via `python -m build --wheel --no-isolation`, with
   `MXNET_PACKAGE_VERSION=<version>` and `MXNET_SETUP_ENABLE_OPENCV_DEPS`
   / `MXNET_SETUP_ENABLE_CUDA_DEPS` toggling the `install_requires` lists
   in `setup.py`.
8. **Validate provenance** (§7) — non-zero exit means *don't ship*.

Knobs:

| Env var | Default | Meaning |
|---------|---------|---------|
| `MXNET_BUILD_JOBS` | nproc | Ninja parallelism |
| `PYTHON` | `.venv-mxnet/bin/python` | interpreter for `-m build` and provenance |
| `BUNDLE_OPENCV` | `1` | copy system OpenCV into the wheel; set `0` only for a deliberate OpenCV-off wheel |
| `MXNET_PACKAGE_VERSION` | today's date | overridden by the positional `<version>` arg |

---

## 5. Versioning and the `<version>` argument

Wheel versions are `2.0.0+cu13.bw.<YYYYMMDD>[.<build>]`. The trailing
`.<build>` disambiguates multiple wheels on the same day — **always pass
it explicitly when rebuilding a date that already has a published wheel**,
otherwise the new wheel collides with the old tag. `tools/update_mxnet_wheel.py`
(used by the d2l side) sorts by `(date, build)`, so `…20260529.1` correctly
supersedes `…20260529`.

```bash
tools/build_cleanup_wheel.sh 2.0.0+cu13.bw.20260529.1
```

---

## 6. Runtime dependency model (what the wheel does *not* bundle)

The wheel is deliberately **not** a 2 GB monolith. Only OpenCV is bundled
inside it. Everything else is reached at load time via RUNPATH:

- **CUDA / cuDNN / NCCL** are declared as pip deps
  (`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`) and resolved from
  `site-packages/nvidia/<pkg>/lib/` — the same layout PyTorch and JAX use.
  The base CUDA runtime (`libcudart`, `libcublas`, …) comes from the
  system toolkit at `/usr/local/cuda/`. (As of this writing NVIDIA ships
  only `nvidia-cudnn-cu13` and `nvidia-nccl-cu13` on PyPI for cu13; the
  others are placeholder stubs, hence the system-toolkit fallback.)
- **OpenBLAS** comes from the `scipy-openblas32` wheel under
  `site-packages/scipy_openblas32/lib/`.
- **OpenCV** is the exception: native `libopencv_*.so` SONAMEs cannot be
  installed reliably via Python metadata, so they are *bundled* into
  `mxnet/lib/` and found via `$ORIGIN/lib`. The wheel is therefore
  self-contained for image I/O and does **not** require the consumer to
  `pip install opencv-python`.

This is why `.venv-mxnet` can run MXNet image notebooks with **no `cv2`
module installed** — the C++ OpenCV is inside the wheel, and `mx.image`
calls it through the C API.

---

## 6a. cuBLAS / driver compatibility (the `NOT_INITIALIZED` trap)

The wheel pins `nvidia-cublas>=13.0,<13.2` in `setup.py`. This is not
cosmetic. `nvidia-cudnn-cu13` declares an **unversioned** `nvidia-cublas`
dependency, so a bare resolve installs the newest cuBLAS (13.4.x / 13.5.x
at time of writing). Those **CUDA 13.2+ cuBLAS builds fail to load their
large-GEMM kernels on the CUDA 13.0 driver line (R580)**:

```
Cublas: Check failed: err == CUBLAS_STATUS_SUCCESS (1 vs. 0) : Sgemm fail
```

The tell is **size-dependent** failure: tiny GEMMs (N≤16) and convolutions
succeed, but any non-trivial `mx.nd.dot` / `np.dot` / `FullyConnected`
(and conv classifiers like AlexNet/NIN) return
`CUBLAS_STATUS_NOT_INITIALIZED` (status `1`). It is **not** an API problem
(swapping `cublasSgemm`→`cublasSgemmEx` does not help), not a workspace
problem (`CUBLAS_WORKSPACE_CONFIG` / `CUDA_MODULE_LOADING=EAGER` /
`MXNET_USE_CUBLASLT=1` do not help), and not an MXNet bug — PyTorch with
its bundled CUDA 12.8 cuBLAS does the same GEMM fine on the same GPU.

Pin to the 13.0/13.1 cuBLAS generation (`13.1.1.3` ships with the CUDA
13.0.3 toolkit and is what the wheel is built against). An older cuBLAS
also runs on newer drivers (forward compatible), so the pin is safe across
the R580 → R590+ range. The alternative fix is upgrading the driver to the
R590+ line that matches a 13.2+ cuBLAS.

To diagnose which cuBLAS is actually loaded:

```bash
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stdout python -c \
  "import mxnet as mx; mx.nd.dot(mx.nd.ones((256,256),ctx=mx.gpu(0)),
   mx.nd.ones((256,256),ctx=mx.gpu(0))).wait_to_read()" 2>&1 | grep -m1 'cuBLAS (v'
# prints e.g. "cuBLAS (v13.1.1) ..."  -- if it says v13.4/v13.5 on an
# R580 host, that is the bug.
```

## 7. The provenance gate — why a green build is trustworthy

`tools/release_provenance.py <wheel> --cmake-cache build/CMakeCache.txt
--package-version <v> --expect-cuda on --expect-cudnn on --expect-nccl on
--expect-onednn on --expect-opencv on` performs **read-only** checks and
exits non-zero on any mismatch. For OpenCV (`--expect-opencv on`) it
asserts all three of:

1. `libmxnet.so` has `libopencv_*` entries in its `NEEDED` list (i.e. it
   was actually *compiled* against OpenCV — this is the check that the
   regression in §1 would have failed);
2. the wheel bundles `mxnet/lib/libopencv_*`;
3. every `NEEDED` `libopencv_*` SONAME is present among the bundled files.

It also checks the package version matches and that the binary's embedded
commit stamp corresponds to the checkout. Treat a non-zero exit as a hard
stop.

---

## 8. Test before you ship

Building is necessary but not sufficient. Run the suite against the
freshly built wheel installed into a real environment:

```bash
uv pip install --python .venv-mxnet/bin/python --force-reinstall dist/mxnet-*.whl
uv pip install --python .venv-mxnet/bin/python pytest pytest-timeout flaky

# OpenCV regression (must NOT skip — skipping means opencv-off):
.venv-mxnet/bin/python -m pytest -q \
  tests/python/unittest/test_d2l_opencv_image_io_regression.py

# d2l regression set + core unit tests
.venv-mxnet/bin/python -m pytest -q tests/python/unittest -k "d2l or image or opencv"

# GPU smoke
.venv-mxnet/bin/python -m pytest -q tests/python/gpu -k "d2l or image or convolution"
```

A quick manual capability check:

```python
import mxnet as mx
from mxnet.runtime import Features
assert Features().is_enabled('OPENCV'), "OpenCV not compiled in!"
print(mx.np.ones((2, 2), ctx=mx.gpu(0)) + 1)   # GPU works
```

---

## 9. Release

```bash
# tag the source the wheel was built from
git tag -a v2.0.0+cu13.bw.20260529.1 -m "CUDA 13 Ampere→Blackwell wheel, OpenCV on"
git push origin master --tags

# publish the wheel as a release asset
gh release create v2.0.0+cu13.bw.20260529.1 dist/mxnet-*.whl \
  --title "v2.0.0+cu13.bw.20260529.1" \
  --notes "Ampere→Blackwell (sm_80/86/89/90/100/120+PTX), CUDA 13, OpenCV on."
```

Downstreams (e.g. d2l) then bump their pin with
`tools/update_mxnet_wheel.py --source github`.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Build with USE_OPENCV=1 for image io` at runtime | wheel compiled OpenCV-off | rebuild with `USE_OPENCV=ON`; the provenance gate should have caught this |
| `does not contain a CMakeLists.txt` at `3rdparty/...` | submodules not initialised | `git submodule update --init --recursive` |
| provenance: `libmxnet.so has no libopencv_* NEEDED` | `libopencv-dev` missing at configure time | install it, **delete `build/`**, reconfigure (a stale cache won't pick it up) |
| provenance: `does not bundle mxnet/lib/libopencv_*` | `BUNDLE_OPENCV=0` or bundling failed | unset `BUNDLE_OPENCV`; ensure system OpenCV `.so` files resolve via `ldconfig -p` |
| `cudaErrorNoKernelImageForDevice` on GPU op | wheel lacks SASS for that GPU | add the arch to `MXNET_CUDA_ARCH`, delete `build/`, rebuild |
| NCCL/cuDNN feature off despite `-DUSE_*=ON` | `-dev` package missing | install `libnccl-dev` / `libcudnn9-dev-cuda-13`, delete `build/`, reconfigure |
| wheel version collides with a published tag | rebuilt same date without `.<build>` | pass an explicit `…<YYYYMMDD>.<N>` version |

> When in doubt about a stale CMake cache, `rm -rf build/` and redo the
> first-time configure (§3). The reconfigure inside the script only flips
> the feature toggles; it does not repair a cache that cached the wrong
> answer for a library that wasn't installed yet.
