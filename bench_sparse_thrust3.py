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
"""Benchmark dense_to_csr, csr_to_dense, and sparse topk under CUDA 13 / Thrust 3.

Measures mean, median, and p99 latency over 10 runs after 3 warmup runs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_sparse_thrust3.py
"""

import time
import numpy as np
import mxnet as mx


WARMUP = 3
ITERS  = 10

DENSE_SHAPES   = [(1024, 8192), (4096, 16384), (16384, 65536)]
DENSITIES      = [0.01, 0.1, 0.5]
TOPK_SHAPES    = [(1024, 8192), (4096, 16384)]
TOPK_KS        = [10, 100, 1000]


def timed_runs(fn, warmup=WARMUP, iters=ITERS):
    """Run fn() warmup times, then iters timed times. Return list of ms durations."""
    for _ in range(warmup):
        fn()
        mx.nd.waitall()

    times_ms = []
    for _ in range(iters):
        mx.nd.waitall()
        t0 = time.perf_counter()
        fn()
        mx.nd.waitall()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
    return times_ms


def stats(times_ms):
    a = np.array(times_ms)
    return float(np.mean(a)), float(np.median(a)), float(np.percentile(a, 99))


def make_dense(shape, density, ctx):
    """Create a dense GPU array with approximately `density` fraction nonzeros."""
    rows, cols = shape
    rng = np.random.default_rng(42)
    mask = rng.random(shape).astype('float32')
    mask[mask >= density] = 0.0
    # nonzero values drawn from uniform [0.1, 1.0] to avoid FP zero coincidence
    vals = rng.uniform(0.1, 1.0, shape).astype('float32')
    data = (mask < density).astype('float32') * vals
    return mx.nd.array(data, ctx=ctx)


def bench_dense_to_csr(ctx):
    print("\n=== dense -> CSR ===")
    print(f"{'shape':>20}  {'density':>7}  {'mean ms':>9}  {'med ms':>8}  {'p99 ms':>8}")
    results = []
    for shape in DENSE_SHAPES:
        for density in DENSITIES:
            dense = make_dense(shape, density, ctx)
            dense.wait_to_read()

            def fn():
                return dense.tostype('csr')

            times = timed_runs(fn)
            mean_ms, med_ms, p99_ms = stats(times)
            row = dict(op='dense_to_csr', shape=shape, density=density,
                       mean_ms=mean_ms, med_ms=med_ms, p99_ms=p99_ms)
            results.append(row)
            shape_str = f"{shape[0]}x{shape[1]}"
            print(f"  {shape_str:>18}  {density:>7.2f}  {mean_ms:>9.3f}  {med_ms:>8.3f}  {p99_ms:>8.3f}")
    return results


def bench_csr_to_dense(ctx):
    # 16384x65536 at fp32 requires ~4 GB dense + CSR overhead; skip to avoid OOM.
    CSR_TO_DENSE_SHAPES = [s for s in DENSE_SHAPES if s[0] * s[1] <= 4096 * 16384]
    print("\n=== CSR -> dense ===")
    print(f"  (skipping 16384x65536 — dense buffer ~4 GB, would OOM)")
    print(f"{'shape':>20}  {'density':>7}  {'mean ms':>9}  {'med ms':>8}  {'p99 ms':>8}")
    results = []
    for shape in CSR_TO_DENSE_SHAPES:
        for density in DENSITIES:
            dense = make_dense(shape, density, ctx)
            dense.wait_to_read()
            csr = dense.tostype('csr')
            csr.wait_to_read()

            def fn(c=csr):
                return c.tostype('default')

            times = timed_runs(fn)
            mean_ms, med_ms, p99_ms = stats(times)
            row = dict(op='csr_to_dense', shape=shape, density=density,
                       mean_ms=mean_ms, med_ms=med_ms, p99_ms=p99_ms)
            results.append(row)
            shape_str = f"{shape[0]}x{shape[1]}"
            print(f"  {shape_str:>18}  {density:>7.2f}  {mean_ms:>9.3f}  {med_ms:>8.3f}  {p99_ms:>8.3f}")
    return results


def bench_topk(ctx):
    print("\n=== topk (dense, axis=-1) ===")
    print(f"{'shape':>20}  {'K':>6}  {'mean ms':>9}  {'med ms':>8}  {'p99 ms':>8}")
    results = []
    for shape in TOPK_SHAPES:
        for k in TOPK_KS:
            if k > shape[-1]:
                continue
            rng = np.random.default_rng(7)
            data = rng.random(shape).astype('float32')
            x = mx.nd.array(data, ctx=ctx)
            x.wait_to_read()

            def fn(arr=x, kk=k):
                return mx.nd.topk(arr, k=kk, axis=-1, ret_typ='indices')

            times = timed_runs(fn)
            mean_ms, med_ms, p99_ms = stats(times)
            row = dict(op='topk', shape=shape, k=k,
                       mean_ms=mean_ms, med_ms=med_ms, p99_ms=p99_ms)
            results.append(row)
            shape_str = f"{shape[0]}x{shape[1]}"
            print(f"  {shape_str:>18}  {k:>6}  {mean_ms:>9.3f}  {med_ms:>8.3f}  {p99_ms:>8.3f}")
    return results


def main():
    ctx = mx.gpu(0)
    # confirm device
    probe = mx.nd.ones((4,), ctx=ctx)
    probe.wait_to_read()
    dev_name = mx.context.gpu_memory_info(0)
    print(f"Device: GPU 0  (CUDA context confirmed)")
    print(f"Warmup={WARMUP}  Iters={ITERS}")

    all_results = []
    all_results += bench_dense_to_csr(ctx)
    all_results += bench_csr_to_dense(ctx)
    all_results += bench_topk(ctx)

    # Machine-readable summary for the markdown table
    print("\n\n=== CSV (for table) ===")
    print("op,shape,density_or_k,mean_ms,med_ms,p99_ms")
    for r in all_results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        if r['op'] == 'topk':
            param = str(r['k'])
        else:
            param = str(r['density'])
        print(f"{r['op']},{shape_str},{param},{r['mean_ms']:.3f},{r['med_ms']:.3f},{r['p99_ms']:.3f}")


if __name__ == '__main__':
    main()
