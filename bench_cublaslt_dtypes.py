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
"""Compare TFLOPS of mx.nd.dot GEMM at 4096^3 under MXNET_USE_CUBLASLT=0
vs =1 across {float16, float32, float64} - the dtypes PR-B added wrappers
for. bf16 still depends on plumbing inside mshadow's dispatcher (PR-C+)
so it is not benched here.

Run with CUDA_VISIBLE_DEVICES=1 to use the second GPU.

Usage:
    python bench_cublaslt_dtypes.py            # runs driver mode automatically
    python bench_cublaslt_dtypes.py --child    # single in-proc child
"""

import argparse
import os
import subprocess
import sys
import time


SHAPES = [
    (4096, 4096, 4096),
]
DTYPES = ['float16', 'float32', 'float64']
WARMUP = 3
ITERS  = 10


def run_inner():
    import mxnet as mx
    print(f"[child] MXNET_USE_CUBLASLT={os.environ.get('MXNET_USE_CUBLASLT', '<unset>')}")
    ctx = mx.gpu(0)
    for dtype in DTYPES:
        for m, n, k in SHAPES:
            a = mx.nd.ones((m, k), ctx=ctx, dtype=dtype)
            b = mx.nd.ones((k, n), ctx=ctx, dtype=dtype)
            # warmup
            for _ in range(WARMUP):
                c = mx.nd.dot(a, b)
                c.wait_to_read()
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
            print(f"  {dtype:>7}  {m:>5}x{n:<5}x{k:<5}  "
                  f"{secs_per*1e3:8.3f} ms  {tflops:8.2f} TFLOPS")


def driver():
    here = os.path.abspath(__file__)
    blobs = {}
    for use_lt in ('0', '1'):
        env = os.environ.copy()
        env['MXNET_USE_CUBLASLT'] = use_lt
        print(f"\n==== MXNET_USE_CUBLASLT={use_lt} ====")
        r = subprocess.run([sys.executable, here, '--child'], env=env,
                           check=True, capture_output=True, text=True)
        blobs[use_lt] = r.stdout
        print(r.stdout, end='')

    import re
    def parse(blob):
        d = {}
        for line in blob.splitlines():
            mline = re.match(
                r'\s*(\w+)\s+(\d+)x(\d+)x(\d+)\s+\S+\s+ms\s+([0-9.]+)\s+TFLOPS', line)
            if mline:
                d[(mline[1], int(mline[2]), int(mline[3]), int(mline[4]))] = float(mline[5])
        return d
    a = parse(blobs['0'])
    b = parse(blobs['1'])
    print("\n==== Summary ====")
    print("dtype     Shape              | legacy TFLOPS | LT TFLOPS | speedup")
    for dtype in DTYPES:
        for s in SHAPES:
            key = (dtype, s[0], s[1], s[2])
            la = a.get(key, float('nan'))
            lb = b.get(key, float('nan'))
            sp = (lb / la) if la == la and la > 0 else float('nan')
            print(f"  {dtype:>7} {s[0]}x{s[1]}x{s[2]:<5}    "
                  f"{la:8.2f}      {lb:8.2f}    {sp:.2f}x")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--driver', action='store_true')
    ap.add_argument('--child', action='store_true')
    args = ap.parse_args()
    if args.child:
        run_inner()
    else:
        driver()
