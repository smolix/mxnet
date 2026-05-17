#!/usr/bin/env python3
"""Sweep a few conv shapes to compare cuDNN versions.

The single TF32-audit shape (3x3, 256->256, 28x28, bs=32) was already
well-tuned at 9.14; sm_120 fallback gaps are more likely on uncommon
sizes. Sample a few:
  - The audit shape (positive control: should be unchanged ~41 TFLOPS)
  - 1x1 conv 512->2048 (common ResNet bottleneck expansion)
  - 7x7 conv 3->64 stride 2 (input conv; memory-bound)
  - Depthwise 3x3 256->256 (group=256)
  - 3x3 grouped conv 256->256 group=32 (cuDNN often falls back here)

Run after each cuDNN swap and compare TFLOPS.
"""
import os
import time
import sys
import mxnet as mx
from mxnet import nd


def bench_conv(ctx, N, C, H, W, K, kh, kw, pad, stride, group, dtype='float32',
               warmup=20, iters=100):
    x = nd.random.uniform(-1, 1, shape=(N, C, H, W), dtype=dtype, ctx=ctx)
    w = nd.random.uniform(-1, 1, shape=(K, C // group, kh, kw), dtype=dtype,
                          ctx=ctx)
    kwargs = dict(data=x, weight=w, kernel=(kh, kw), pad=pad, stride=stride,
                  num_filter=K, num_group=group, no_bias=True)
    for _ in range(warmup):
        y = nd.Convolution(**kwargs)
    y.wait_to_read()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = nd.Convolution(**kwargs)
    y.wait_to_read()
    t1 = time.perf_counter()
    per_iter_ms = (t1 - t0) / iters * 1000.0
    out_h = y.shape[2]
    out_w = y.shape[3]
    flops = 2.0 * N * K * out_h * out_w * (C // group) * kh * kw
    tflops = flops / (per_iter_ms / 1000.0) / 1e12
    return per_iter_ms, tflops


SHAPES = [
    # name, N, C, H, W, K, kh, kw, pad, stride, group
    ('audit_3x3_28x28_256-256_bs32',  32, 256, 28, 28, 256, 3, 3, (1,1), (1,1), 1),
    ('1x1_14x14_512-2048_bs32',       32, 512, 14, 14, 2048,1, 1, (0,0), (1,1), 1),
    ('7x7_224x224_3-64_stride2_bs32', 32, 3,  224,224, 64, 7, 7, (3,3), (2,2), 1),
    ('dw_3x3_56x56_256-256_g256_bs32',32,256, 56, 56, 256, 3, 3, (1,1), (1,1), 256),
    ('gp_3x3_28x28_256-256_g32_bs32', 32, 256, 28, 28, 256, 3, 3, (1,1), (1,1), 32),
]


def main():
    ctx = mx.gpu(0)
    print(f'{"shape":<40s} {"ms":>10s} {"TFLOPS":>10s}')
    print('-' * 64)
    for row in SHAPES:
        name = row[0]
        try:
            ms, tf = bench_conv(ctx, *row[1:])
            print(f'{name:<40s} {ms:>10.3f} {tf:>10.2f}')
        except Exception as e:
            print(f'{name:<40s}  ERROR: {e}')


if __name__ == '__main__':
    main()
