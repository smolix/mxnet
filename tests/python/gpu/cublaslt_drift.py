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
"""Quantify cuBLASLt-vs-legacy gemm numerical drift.

Backend is chosen by MXNET_USE_CUBLASLT (read once at import). This script runs
a fixed, seeded set of gemms (Dense/FC + dot) for a backend and writes the
results to an .npz. Run it once per backend, then compare the two .npz files.

Usage: cublaslt_drift.py <out.npz> <gpu_id>
"""
import sys
import numpy as onp
import mxnet as mx
from mxnet import np, npx
npx.set_np()


def main():
    out_path = sys.argv[1]
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dev = mx.gpu(gpu)

    results = {}
    # Deterministic inputs via numpy seed, then move to device.
    rng = onp.random.RandomState(0)
    shapes = [(64, 64, 64), (128, 512, 512), (8, 4096, 1024),
              (256, 256, 1000), (1, 2048, 2048), (333, 777, 511)]
    for dt in ["float32", "float16"]:
        for (m, k, n) in shapes:
            a = np.array(rng.randn(m, k).astype("float32"), device=dev, dtype=dt)
            b = np.array(rng.randn(k, n).astype("float32"), device=dev, dtype=dt)
            c = np.dot(a, b)
            c.wait_to_read()
            results[f"dot_{dt}_{m}_{k}_{n}"] = c.asnumpy().astype("float32")
        # FC via fully_connected (the op that gets captured).
        for (m, k, n) in shapes:
            x = np.array(rng.randn(m, k).astype("float32"), device=dev, dtype=dt)
            w = np.array(rng.randn(n, k).astype("float32"), device=dev, dtype=dt)
            y = npx.fully_connected(x, w, no_bias=True, num_hidden=n, flatten=True)
            y.wait_to_read()
            results[f"fc_{dt}_{m}_{k}_{n}"] = y.asnumpy().astype("float32")

    npx.waitall()
    onp.savez(out_path, **results)
    print(f"wrote {out_path} ({len(results)} tensors)")


if __name__ == "__main__":
    main()
