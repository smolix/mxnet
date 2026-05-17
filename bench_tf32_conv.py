#!/usr/bin/env python3
"""Reproduce the TF32 audit benchmark: 3x3 conv, batch 32, 28x28, 256->256.

Run with:
  MXNET_CUDNN_AUTOTUNE_DEFAULT=2 python bench_tf32_conv.py
"""
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd


def bench(ctx, dtype, warmup=30, iters=200):
    N, C, H, W, K = 32, 256, 28, 28, 256
    x = nd.random.uniform(-1, 1, shape=(N, C, H, W), dtype=dtype, ctx=ctx)
    w = nd.random.uniform(-1, 1, shape=(K, C, 3, 3), dtype=dtype, ctx=ctx)

    # Warmup
    for _ in range(warmup):
        y = nd.Convolution(
            data=x, weight=w, kernel=(3, 3), pad=(1, 1),
            num_filter=K, no_bias=True,
        )
    y.wait_to_read()

    # Timed
    t0 = time.perf_counter()
    for _ in range(iters):
        y = nd.Convolution(
            data=x, weight=w, kernel=(3, 3), pad=(1, 1),
            num_filter=K, no_bias=True,
        )
    y.wait_to_read()
    t1 = time.perf_counter()

    per_iter_ms = (t1 - t0) / iters * 1000.0
    # 2 * N * K * H * W * C * Kh * Kw FLOPS
    flops_per_iter = 2.0 * N * K * H * W * C * 3 * 3
    tflops = flops_per_iter / (per_iter_ms / 1000.0) / 1e12
    out_sum = float(y.sum().asscalar())
    return per_iter_ms, tflops, out_sum


def main():
    ctx = mx.gpu(0)
    print(f'cuDNN bound version: {os.popen("strings " + mx.__file__.replace("__init__.py","libmxnet.so") + " 2>/dev/null | grep cuDNN | head").read().strip()[:200]}')
    per_iter, tflops, sum_y = bench(ctx, 'float32')
    print(f'3x3 conv 28x28 256->256 bs=32 FP32  per_iter={per_iter:.3f} ms  TFLOPS={tflops:.2f}  out_sum={sum_y:.4f}')


if __name__ == '__main__':
    main()
