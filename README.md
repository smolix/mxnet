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

Blackwell / CUDA 13 fork
========================

> Unofficial fork at [`smolix/mxnet`](https://github.com/smolix/mxnet) — Blackwell
> (sm_120) port of MXNet 2.0.
>
> Apache MXNet was **archived on 2023-11-17**. The upstream tree is frozen at
> CUDA 11 / cuDNN 8 / oneDNN v2 and does not build on Blackwell GPUs or modern
> CUDA toolchains. This fork carries the minimum set of patches needed to run
> existing MXNet code on current hardware. It is not an official Apache release.

Current version string: `2.0.0+cu13.bw.20260517`
(`<upstream-version>+cu<cuda-major>.bw.<YYYYMMDD>`).

Why this fork exists
--------------------

The primary goal is to **run existing MXNet notebooks on Blackwell** (RTX PRO
4000 / RTX 50-series / B100-class) hardware with the current CUDA 13 +
cuDNN 9 + oneDNN v3 + NCCL 2.28 stack. The secondary goal is to keep the
residual MXNet user community (legacy research code, frozen production
pipelines, niche operators like `_contrib_quantize_*`) able to use current
GPUs without a full rewrite to PyTorch / JAX. See [`issues.md`](issues.md) for
the open work list and priority-ordered triage.

What works
----------

* Blackwell `sm_120` SASS / PTX (CUDA 13.0), plus sm_80 (Ampere), sm_86
  (Ada), sm_89 (RTX 40), sm_90 (H100), and PTX 120 fallback in fatbin.
* cuDNN 9.22 — including the rewritten v8 RNN path (LSTM / GRU / vanilla
  RNN, fwd + bwd). TF32 enabled by default on FP32 conv (mirrors PyTorch /
  TensorFlow defaults; ~2.87× speedup on sm_120 vs the legacy
  `MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION=0` mode).
* oneDNN v3.11 — full INT8 path (per-OC weight scales, fused conv/FC, fused
  sum, dequant-to-fp32 output).
* NCCL 2.28 — single-process / multi-GPU.
* INT8 quantization (`quantize_net`, `_sg_onednn_conv`, `_sg_onednn_fully_connected`).
* fp16 and fp32 forward + backward training.
* F16C CPU intrinsics for fast fp16 host (de)serialization.
* DNNL subgraph fusion, the activation/eltwise/layer-norm/softmax stack,
  pooling, batch norm fwd+bwd, transpose, concat, where, masked softmax.

What is experimental or known-broken
------------------------------------

* **bf16 on non-AVX-512-BF16 CPUs** — oneDNN falls back to fp32 emulation;
  not fixable in software, test on Intel SPR or AMD Zen 4 / Granite Rapids.
* **Backward through quantized ops** — forward inference is solid; backward
  through `_sg_onednn_fully_connected` and `_sg_onednn_conv` is unvalidated.
* **AMP (automatic mixed precision) subgraph** — 6 known failures with
  `inner_product` primitive creation; investigation pending.
* **int8 quantized concat** — `test_pos_single_concat_pos_neg[*-data_shape1]`
  fails with entire output channels zeroed; suspect oneDNN v3 uint8→int8
  reorder semantics.
* **ONNX export / import** — both `tests/python/onnx/test_models.py` and
  `test_operators.py` error at collect time; the ONNX path was not updated
  for MXNet 2.0 numpy ops.
* See [`issues.md`](issues.md) for the full open list (45 items).

System requirements
-------------------

* Linux x86_64 (tested on Ubuntu 22.04 / 24.04).
* NVIDIA driver supporting CUDA 13 (R570+).
* CUDA 13.0 toolkit.
* cuDNN **9.22+** (cuDNN 9.22 has the best `sm_120` heuristic coverage;
  earlier 9.x works but routes more conv shapes through generic fallback
  engines — notably depthwise 3×3 is ~7× faster on 9.22 vs 9.14). The
  release wheel bundles 9.22 under `mxnet/lib/`.
* NCCL 2.28.3.
* Python 3.10+ (3.11 / 3.12 / 3.13 are CI-tested).

Installation
------------

```bash
pip install https://github.com/smolix/mxnet/releases/download/v2.0.0.cu13.bw.20260517-beta/mxnet-2.0.0+cu13.bw.20260517-cp311-cp311-linux_x86_64.whl
```

The wheel itself is **454 MB**. `pip` will transitively pull
`nvidia-cudnn-cu13` (~1 GB) and `nvidia-nccl-cu13` (~190 MB). The
remaining CUDA 13 toolkit libs (`libcudart.so.13`, `libcublas.so.13`,
`libcufft.so.12`, `libcusolver.so.12`, `libcurand.so.10`,
`libnvrtc.so.13`) come from your system CUDA 13 install at
`/usr/local/cuda/` — NVIDIA has not yet published `cu13` wheels for
those on PyPI (the rest of the `nvidia-*-cu13` packages are placeholder
stubs at version `0.0.1` as of 2026-05-17). `libmxnet.so`'s `RUNPATH`
covers both locations.

Requires Python 3.11, Linux x86_64, NVIDIA driver R570+, and the CUDA
13 toolkit installed at `/usr/local/cuda/` (e.g. `apt install cuda-13`).

To **build from source** see [`BUILDING.md`](BUILDING.md). The short version
is: clone with submodules, install `libnccl-dev` *before* invoking `cmake`,
then `cmake -DUSE_CUDA=ON -DCUDA_ARCH_LIST="12.0" ..`.

Acknowledgements
----------------

This fork builds on the work of the Apache MXNet community and its
contributors. All upstream code is Apache 2.0; the Blackwell / CUDA 13
patches in this fork are likewise Apache 2.0. The original project history
follows below — its build status badges, social links, and roadmap targets
refer to the (now archived) upstream and are kept for historical reference.

---

<div align="center">
  <a href="https://mxnet.apache.org/"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet_logo_2.png"></a><br>
</div>

[![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)](https://mxnet.apache.org)

Apache MXNet for Deep Learning (upstream — archived 2023-11-17)
================================================================

> **Note:** the sections below describe the original Apache MXNet 2.0
> project. They are kept verbatim for historical context. Build status
> badges, mailing lists, Slack channels, and Twitter/Medium links refer to
> the **archived** upstream project and are not actively monitored.
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/apache/mxnet)](https://github.com/apache/mxnet/releases) [![GitHub stars](https://img.shields.io/github/stars/apache/mxnet)](https://github.com/apache/mxnet/stargazers) [![GitHub forks](https://img.shields.io/github/forks/apache/mxnet)](https://github.com/apache/mxnet/network) [![GitHub contributors](https://img.shields.io/github/contributors-anon/apache/mxnet)](https://github.com/apache/mxnet/graphs/contributors) [![GitHub issues](https://img.shields.io/github/issues/apache/mxnet)](https://github.com/apache/mxnet/issues) [![good first issue](https://img.shields.io/github/issues/apache/mxnet/good%20first%20issue)](https://github.com/apache/mxnet/labels/good%20first%20issue) [![GitHub pull requests by-label](https://img.shields.io/github/issues-pr/apache/mxnet/pr-awaiting-review)](https://github.com/apache/mxnet/labels/pr-awaiting-review) [![GitHub license](https://img.shields.io/github/license/apache/mxnet)](https://github.com/apache/mxnet/blob/master/LICENSE) [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fapache%2Fmxnet)](https://twitter.com/intent/tweet?text=Wow:%20https%3A%2F%2Fgithub.com%2Fapache%2Fmxnet%20@ApacheMXNet) [![Twitter Follow](https://img.shields.io/twitter/follow/ApacheMXNet?style=social)](https://twitter.com/ApacheMXNet)

Apache MXNet is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** [symbolic and imperative programming](https://mxnet.apache.org/api/architecture/program_model)
to ***maximize*** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scalable to many GPUs and machines.

Apache MXNet is more than a deep learning project. It is a [community](https://mxnet.apache.org/versions/master/community)
on a mission of democratizing AI. It is a collection of [blue prints and guidelines](https://mxnet.apache.org/api/architecture/overview)
for building deep learning systems, and interesting insights of DL systems for hackers.

Licensed under an [Apache-2.0](https://github.com/apache/mxnet/blob/master/LICENSE) license.

| Branch  | Build Status  |
|:-------:|:-------------:|
| [master](https://github.com/apache/mxnet/tree/master) | [![CentOS CPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/centos-cpu/job/master/badge/icon?subject=build%20centos%20cpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/centos-cpu/job/master/) [![CentOS GPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/centos-gpu/job/master/badge/icon?subject=build%20centos%20gpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/centos-gpu/job/master/) [![Clang Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/clang/job/master/badge/icon?subject=build%20clang)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/clang/job/master/) <br> [![Edge Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/edge/job/master/badge/icon?subject=build%20edge)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/edge/job/master/) [![Miscellaneous Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/miscellaneous/job/master/badge/icon?subject=build%20miscellaneous)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/miscellaneous/job/master/) [![Sanity Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/sanity/job/master/badge/icon?subject=build%20sanity)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/sanity/job/master/) <br> [![Unix CPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/unix-cpu/job/master/badge/icon?subject=build%20unix%20cpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/unix-cpu/job/master/) [![Unix GPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/unix-gpu/job/master/badge/icon?subject=build%20unix%20gpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/unix-gpu/job/master/) [![Website Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/website/job/master/badge/icon?subject=build%20website)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/website/job/master/) <br> [![Windows CPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/windows-cpu/job/master/badge/icon?subject=build%20windows%20cpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/windows-cpu/job/master/) [![Windows GPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/windows-gpu/job/master/badge/icon?subject=build%20windows%20gpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/windows-gpu/job/master/) [![Documentation Status](http://jenkins.mxnet-ci.com/job/restricted-website-build/badge/icon)](https://mxnet.apache.org/) |
| [v1.x](https://github.com/apache/mxnet/tree/v1.x) | [![CentOS CPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/centos-cpu/job/v1.x/badge/icon?subject=build%20centos%20cpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/centos-cpu/job/v1.x/) [![CentOS GPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/centos-gpu/job/v1.x/badge/icon?subject=build%20centos%20gpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/centos-gpu/job/v1.x/) [![Clang Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/clang/job/v1.x/badge/icon?subject=build%20clang)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/clang/job/v1.x/) <br> [![Edge Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/edge/job/v1.x/badge/icon?subject=build%20edge)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/edge/job/v1.x/) [![Miscellaneous Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/miscellaneous/job/v1.x/badge/icon?subject=build%20miscellaneous)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/miscellaneous/job/v1.x/) [![Sanity Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/sanity/job/v1.x/badge/icon?subject=build%20sanity)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/sanity/job/v1.x/) <br> [![Unix CPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/unix-cpu/job/v1.x/badge/icon?subject=build%20unix%20cpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/unix-cpu/job/v1.x/) [![Unix GPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/unix-gpu/job/v1.x/badge/icon?subject=build%20unix%20gpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/unix-gpu/job/v1.x/) [![Website Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/website/job/v1.x/badge/icon?subject=build%20website)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/website/job/v1.x/) <br> [![Windows CPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/windows-cpu/job/v1.x/badge/icon?subject=build%20windows%20cpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/windows-cpu/job/v1.x/) [![Windows GPU Build Status](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/windows-gpu/job/v1.x/badge/icon?subject=build%20windows%20gpu)](http://jenkins.mxnet-ci.com/job/mxnet-validation/job/windows-gpu/job/v1.x/) [![Documentation Status](http://jenkins.mxnet-ci.com/job/restricted-website-build/badge/icon)](https://mxnet.apache.org/) |

Features
--------
* NumPy-like programming interface, and is integrated with the new, easy-to-use Gluon 2.0 interface. NumPy users can easily adopt MXNet and start in deep learning.
* Automatic hybridization provides imperative programming with the performance of traditional symbolic programming.
* Lightweight, memory-efficient, and portable to smart devices through native cross-compilation support on ARM, and through ecosystem projects such as [TVM](https://tvm.ai), [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html), [OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).
* Scales up to multi GPUs and distributed setting with auto parallelism through [ps-lite](https://github.com/dmlc/ps-lite), [Horovod](https://github.com/horovod/horovod), and [BytePS](https://github.com/bytedance/byteps).
* Extensible backend that supports full customization, allowing integration with custom accelerator libraries and in-house hardware without the need to maintain a fork.
* Support for [Python](https://mxnet.apache.org/api/python), [Java](https://mxnet.apache.org/api/java), [C++](https://mxnet.apache.org/api/cpp), [R](https://mxnet.apache.org/api/r), [Scala](https://mxnet.apache.org/api/scala), [Clojure](https://mxnet.apache.org/api/clojure), [Go](https://github.com/jdeng/gomxnet/), [Javascript](https://github.com/dmlc/mxnet.js/), [Perl](https://mxnet.apache.org/api/perl), and [Julia](https://mxnet.apache.org/api/julia).
* Cloud-friendly and directly compatible with AWS and Azure.

Contents
--------
* [Installation](https://mxnet.apache.org/get_started)
* [Tutorials](https://mxnet.apache.org/api/python/docs/tutorials/)
* [Ecosystem](https://mxnet.apache.org/ecosystem)
* [API Documentation](https://mxnet.apache.org/api)
* [Examples](https://github.com/apache/mxnet-examples)
* [Stay Connected](#stay-connected)
* [Social Media](#social-media)

What's New
----------
* [1.9.1 Release](https://github.com/apache/mxnet/releases/tag/1.9.1) - MXNet 1.9.1 Release.
* [1.8.0 Release](https://github.com/apache/mxnet/releases/tag/1.8.0) - MXNet 1.8.0 Release.
* [1.7.0 Release](https://github.com/apache/mxnet/releases/tag/1.7.0) - MXNet 1.7.0 Release.
* [1.6.0 Release](https://github.com/apache/mxnet/releases/tag/1.6.0) - MXNet 1.6.0 Release.
* [1.5.1 Release](https://github.com/apache/mxnet/releases/tag/1.5.1) - MXNet 1.5.1 Patch Release.
* [1.5.0 Release](https://github.com/apache/mxnet/releases/tag/1.5.0) - MXNet 1.5.0 Release.
* [1.4.1 Release](https://github.com/apache/mxnet/releases/tag/1.4.1) - MXNet 1.4.1 Patch Release.
* [1.4.0 Release](https://github.com/apache/mxnet/releases/tag/1.4.0) - MXNet 1.4.0 Release.
* [1.3.1 Release](https://github.com/apache/mxnet/releases/tag/1.3.1) - MXNet 1.3.1 Patch Release.
* [1.3.0 Release](https://github.com/apache/mxnet/releases/tag/1.3.0) - MXNet 1.3.0 Release.
* [1.2.0 Release](https://github.com/apache/mxnet/releases/tag/1.2.0) - MXNet 1.2.0 Release.
* [1.1.0 Release](https://github.com/apache/mxnet/releases/tag/1.1.0) - MXNet 1.1.0 Release.
* [1.0.0 Release](https://github.com/apache/mxnet/releases/tag/1.0.0) - MXNet 1.0.0 Release.
* [0.12.1 Release](https://github.com/apache/mxnet/releases/tag/0.12.1) - MXNet 0.12.1 Patch Release.
* [0.12.0 Release](https://github.com/apache/mxnet/releases/tag/0.12.0) - MXNet 0.12.0 Release.
* [0.11.0 Release](https://github.com/apache/mxnet/releases/tag/0.11.0) - MXNet 0.11.0 Release.
* [Apache Incubator](http://incubator.apache.org/projects/mxnet.html) - We are now an Apache Incubator project.
* [0.10.0 Release](https://github.com/apache/mxnet/releases/tag/v0.10.0) - MXNet 0.10.0 Release.
* [0.9.3 Release](./docs/architecture/release_note_0_9.md) - First 0.9 official release.
* [0.9.1 Release (NNVM refactor)](./docs/architecture/release_note_0_9.md) - NNVM branch is merged into master now. An official release will be made soon.
* [0.8.0 Release](https://github.com/apache/mxnet/releases/tag/v0.8.0)

### Ecosystem News

* [oneDNN for Faster CPU Performance](docs/python_docs/python/tutorials/performance/backend/dnnl/dnnl_readme.md)
* [MXNet Memory Monger, Training Deeper Nets with Sublinear Memory Cost](https://github.com/dmlc/mxnet-memonger)
* [Tutorial for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)
* [MXNet.js: Javascript Package for Deep Learning in Browser (without server)](https://github.com/dmlc/mxnet.js/)
* [Guide to Creating New Operators (Layers)](https://mxnet.apache.org/api/faq/new_op)
* [Go binding for inference](https://github.com/songtianyi/go-mxnet-predictor)

Stay Connected
--------------

| Channel | Purpose |
|---|---|
| [Follow MXNet Development on Github](https://github.com/apache/mxnet/issues) | See what's going on in the MXNet project. |
| [MXNet Confluence Wiki for Developers](https://cwiki.apache.org/confluence/display/MXNET/Apache+MXNet+Home) <i class="fas fa-external-link-alt"> | MXNet developer wiki for information related to project development, maintained by contributors and developers. To request write access, send an email to [send request to the dev list](mailto:dev@mxnet.apache.org?subject=Requesting%20CWiki%20write%20access) <i class="far fa-envelope"></i>. |
| [dev@mxnet.apache.org mailing list](https://lists.apache.org/list.html?dev@mxnet.apache.org) | The "dev list". Discussions about the development of MXNet. To subscribe, send an email to [dev-subscribe@mxnet.apache.org](mailto:dev-subscribe@mxnet.apache.org) <i class="far fa-envelope"></i>. |
| [discuss.mxnet.io](https://discuss.mxnet.io) <i class="fas fa-external-link-alt"></i> | Asking & answering MXNet usage questions. |
| [Apache Slack #mxnet Channel](https://the-asf.slack.com/archives/C7FN4FCP9) <i class="fas fa-external-link-alt"> | Connect with MXNet and other Apache developers. To join the MXNet slack channel [send request to the dev list](mailto:dev@mxnet.apache.org?subject=Requesting%20slack%20access) <i class="far fa-envelope"></i>. |
| [Follow MXNet on Social Media](#social-media) | Get updates about new features and events. |


### Social Media

Keep connected with the latest MXNet news and updates.

<p>
<a href="https://twitter.com/apachemxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/twitter.svg?sanitize=true" height="30px"/> Apache MXNet on Twitter</a>
</p>
<p>
<a href="https://medium.com/apache-mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/medium_black.svg?sanitize=true" height="30px"/> Contributor and user blogs about MXNet</a>
</p>
<p>
<a href="https://reddit.com/r/mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/reddit_blue.svg?sanitize=true" height="30px" alt="reddit"/> Discuss MXNet on r/mxnet</a>
</p>
<p>
<a href="https://www.youtube.com/apachemxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/youtube_red.svg?sanitize=true" height="30px"/> Apache MXNet YouTube channel</a>
</p>
<p>
<a href="https://www.linkedin.com/company/apache-mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/linkedin.svg?sanitize=true" height="30px"/> Apache MXNet on LinkedIn</a>
</p>


History
-------
MXNet emerged from a collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva), and [purine2](https://github.com/purine/purine2). The project reflects what we have learned from the past projects. MXNet combines aspects of each of these projects to achieve flexibility, speed, and memory efficiency.

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
