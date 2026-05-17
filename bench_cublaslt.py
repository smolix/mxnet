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
"""Compare TFLOPS of mx.nd.dot fp32 GEMM at 1024^3 / 4096^3 / 8192^3 under
MXNET_USE_CUBLASLT=0 vs =1.

Run with CUDA_VISIBLE_DEVICES=1 to use the second GPU.

Usage:
    MXNET_USE_CUBLASLT=0 python bench_cublaslt.py
    MXNET_USE_CUBLASLT=1 python bench_cublaslt.py
    # or run as a driver that spawns both subprocesses:
    python bench_cublaslt.py --driver
"""

import argparse
import os
import subprocess
import sys
import textwrap
import time


SHAPES = [
    (1024, 1024, 1024),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]
WARMUP = 3
ITERS  = 10


def run_inner():
    import mxnet as mx
    print(f"[child] MXNET_USE_CUBLASLT={os.environ.get('MXNET_USE_CUBLASLT', '<unset>')}")
    ctx = mx.gpu(0)
    results = {}
    for m, n, k in SHAPES:
        a = mx.nd.ones((m, k), ctx=ctx, dtype='float32')
        b = mx.nd.ones((k, n), ctx=ctx, dtype='float32')
        # warmup
        for _ in range(WARMUP):
            c = mx.nd.dot(a, b)
            c.wait_to_read()
        # timed
        mx.nd.waitall()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            c = mx.nd.dot(a, b)
        c.wait_to_read()
        mx.nd.waitall()
        t1 = time.perf_counter()
        secs_per = (t1 - t0) / ITERS
        flops    = 2.0 * m * n * k
        tflops   = flops / secs_per / 1e12
        print(f"  {m:>5}x{n:<5}x{k:<5}  {secs_per*1e3:8.3f} ms  {tflops:8.2f} TFLOPS")
        results[(m, n, k)] = tflops
    return results


def driver():
    here = os.path.abspath(__file__)
    by = {}
    for use_lt in ('0', '1'):
        env = os.environ.copy()
        env['MXNET_USE_CUBLASLT'] = use_lt
        print(f"\n==== MXNET_USE_CUBLASLT={use_lt} ====")
        r = subprocess.run([sys.executable, here, '--child'], env=env,
                           check=True, capture_output=True, text=True)
        by[use_lt] = r.stdout
        print(r.stdout, end='')

    import re
    def parse(blob):
        d = {}
        for line in blob.splitlines():
            mline = re.match(r'\s*(\d+)x(\d+)x(\d+)\s+\S+\s+ms\s+([0-9.]+)\s+TFLOPS', line)
            if mline:
                d[(int(mline[1]), int(mline[2]), int(mline[3]))] = float(mline[4])
        return d
    a = parse(by['0'])
    b = parse(by['1'])
    print("\n==== Summary ====")
    print("Shape                | legacy TFLOPS | LT TFLOPS | speedup")
    for s in SHAPES:
        la = a.get(s, float('nan'))
        lb = b.get(s, float('nan'))
        sp = (lb / la) if la and la == la else float('nan')
        print(f"  {s[0]}x{s[1]}x{s[2]:<5}        {la:8.2f}        {lb:8.2f}     {sp:.2f}x")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--driver', action='store_true')
    ap.add_argument('--child', action='store_true')
    args = ap.parse_args()
    if args.driver or (not args.child and 'MXNET_USE_CUBLASLT' not in os.environ):
        driver()
    else:
        run_inner()
