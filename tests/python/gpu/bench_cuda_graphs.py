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
"""CUDA Graphs microbenchmark (Phase 0).

Measures per-iteration wall time graphs-on vs graphs-off for hybridized,
static-shape gluon blocks. Run one config per process (the enable flag is read
when the op segment's CudaGraphsExec is first constructed).

Usage:
    bench_cuda_graphs.py <model> <on|off> [gpu_id] [iters]

model:
    chain   : many small elementwise ops (dispatch-bound; all capturable)
    mlp     : Dense stack (FC -> currently bypassed cuBLAS path)
    convnet : small conv stack (cuDNN; already capturable)
"""
import os
import sys
import time

import mxnet as mx
from mxnet import np, npx, gluon
from mxnet.gluon import nn

npx.set_np()


def build(model):
    net = nn.HybridSequential()
    if model == "chain":
        # 64 tiny elementwise ops on a small tensor -> launch/dispatch bound.
        for _ in range(64):
            net.add(nn.Activation("tanh"))
        shape = (32, 64)
    elif model == "mlp":
        for _ in range(8):
            net.add(nn.Dense(512, activation="relu"))
        shape = (64, 512)
    elif model == "convnet":
        for _ in range(8):
            net.add(nn.Conv2D(channels=32, kernel_size=3, padding=1, activation="relu"))
        shape = (16, 32, 32, 32)
    else:
        raise SystemExit(f"unknown model {model}")
    return net, shape


def main():
    model = sys.argv[1]
    mode = sys.argv[2]
    gpu_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

    os.environ["MXNET_ENABLE_CUDA_GRAPHS"] = "1" if mode == "on" else "0"
    os.environ.setdefault("MXNET_USE_FUSION", "0")

    dev = mx.gpu(gpu_id)
    net, shape = build(model)
    net.initialize(device=dev)
    net.hybridize(static_alloc=True, static_shape=True)

    x = np.random.uniform(size=shape, device=dev)

    # Warm-up: first run is conventional (primes tempspace + cuDNN algos), then a
    # few more so the graph is captured and cached before timing.
    for _ in range(50):
        y = net(x)
    y.wait_to_read()
    npx.waitall()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = net(x)
    y.wait_to_read()
    npx.waitall()
    dt = time.perf_counter() - t0

    us = dt / iters * 1e6
    print(f"{model:8s} graphs={mode:3s} gpu={gpu_id} iters={iters} "
          f"per_iter={us:8.2f}us total={dt:.3f}s")


if __name__ == "__main__":
    main()
