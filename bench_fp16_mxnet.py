#!/usr/bin/env python3
"""MXNet-only fp16 benchmark. Outputs JSON to /workspace/mxnet/fp16_bench_mxnet.json"""
import os, time, json, statistics
import mxnet as mx
import mxnet.numpy as mnp
from mxnet.gluon import nn as gnn

WARMUP = 5
RUNS = 20
GPU_ID = int(os.environ.get("BENCH_GPU", "0"))
mx_ctx = mx.gpu(GPU_ID)

def mx_sync():
    mx.nd.waitall()

def mx_time_fn(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    mx_sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        mx_sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times), statistics.stdev(times)

results = []

def record(kernel, shape_str, ms, std, notes=""):
    results.append(dict(kernel=kernel, shape=shape_str, ms=ms, std=std, notes=notes))
    print(f"  {kernel:12s} {shape_str:35s}  {ms:8.3f} ms  (±{std:.3f})")

# 1. Conv2D - use nd.Convolution directly to avoid Gluon np-ndarray requirement
print("=== Conv2D ===")
from mxnet import nd
import numpy as np

for batch in [32, 64, 128]:
    # Build weight/bias manually for nd.Convolution
    W = nd.random.uniform(shape=(256, 256, 3, 3), dtype="float16", ctx=mx_ctx)
    b = nd.zeros((256,), dtype="float16", ctx=mx_ctx)
    x = nd.random.uniform(shape=(batch, 256, 32, 32), dtype="float16", ctx=mx_ctx)
    mx_sync()
    def _conv():
        nd.Convolution(data=x, weight=W, bias=b, kernel=(3,3), pad=(1,1), num_filter=256)
    ms, std = mx_time_fn(_conv)
    record("Conv2D", f"batch={batch},C=256,HW=32", ms, std)

# 2. Dense/matmul via nd.dot
print("=== Dense ===")
for M, N, K in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]:
    A = nd.random.uniform(shape=(M, K), dtype="float16", ctx=mx_ctx)
    B = nd.random.uniform(shape=(K, N), dtype="float16", ctx=mx_ctx)
    mx_sync()
    ms, std = mx_time_fn(lambda: nd.dot(A, B))
    record("Dense", f"MNK=({M},{N},{K})", ms, std)

# 3. Softmax
print("=== Softmax ===")
for batch, hidden in [(32, 4096), (128, 16384)]:
    x = nd.random.uniform(shape=(batch, hidden), dtype="float16", ctx=mx_ctx)
    mx_sync()
    ms, std = mx_time_fn(lambda: nd.softmax(x, axis=-1))
    record("Softmax", f"({batch},{hidden})", ms, std)

# 4. LayerNorm via nd.LayerNorm
print("=== LayerNorm ===")
for batch, seq, hidden in [(32, 128, 768), (8, 1024, 4096)]:
    # LayerNorm over last axis: gamma, beta shapes (hidden,)
    gamma = nd.ones((hidden,), dtype="float16", ctx=mx_ctx)
    beta  = nd.zeros((hidden,), dtype="float16", ctx=mx_ctx)
    x = nd.random.uniform(shape=(batch, seq, hidden), dtype="float16", ctx=mx_ctx)
    mx_sync()
    # nd.LayerNorm normalizes over axis=-1
    ms, std = mx_time_fn(lambda: nd.LayerNorm(data=x, gamma=gamma, beta=beta, axis=-1))
    record("LayerNorm", f"({batch},{seq},{hidden})", ms, std)

# 5. Elementwise add
print("=== Add ===")
for numel in [1_000_000, 4_000_000, 16_000_000]:
    a = nd.random.uniform(shape=(numel,), dtype="float16", ctx=mx_ctx)
    b = nd.random.uniform(shape=(numel,), dtype="float16", ctx=mx_ctx)
    mx_sync()
    ms, std = mx_time_fn(lambda: a + b)
    record("Add", f"{numel//1_000_000}M", ms, std)

# 6. Elementwise mul
print("=== Mul ===")
for numel in [1_000_000, 4_000_000, 16_000_000]:
    a = nd.random.uniform(shape=(numel,), dtype="float16", ctx=mx_ctx)
    b = nd.random.uniform(shape=(numel,), dtype="float16", ctx=mx_ctx)
    mx_sync()
    ms, std = mx_time_fn(lambda: a * b)
    record("Mul", f"{numel//1_000_000}M", ms, std)

with open("/workspace/mxnet/fp16_bench_mxnet.json", "w") as f:
    json.dump({"framework": "mxnet", "version": mx.__version__,
               "warmup": WARMUP, "runs": RUNS, "results": results}, f, indent=2)
print("Saved /workspace/mxnet/fp16_bench_mxnet.json")
