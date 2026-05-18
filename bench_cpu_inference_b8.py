"""
B8 CPU inference benchmark — apache/mxnet#19218
Measures latency for Conv2D, Dense, and softmax on CPU with MXNet/oneDNN.
Usage:
  python bench_cpu_inference_b8.py              # default OMP_NUM_THREADS
  OMP_NUM_THREADS=1 python bench_cpu_inference_b8.py
  DNNL_VERBOSE=1 python bench_cpu_inference_b8.py
"""
import sys
import os
import time

sys.path.insert(0, '/workspace/mxnet/python')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # force CPU

import mxnet as mx
import mxnet.numpy as mnp
from mxnet import gluon
from mxnet.gluon import nn

CTX = mx.cpu()
WARMUP = 10
RUNS = 50

def bench(model, data, label):
    model.initialize(ctx=CTX)
    # Warmup
    for _ in range(WARMUP):
        out = model(data)
        out.wait_to_read()
    # Timed runs
    t0 = time.perf_counter()
    for _ in range(RUNS):
        out = model(data)
        out.wait_to_read()
    elapsed = (time.perf_counter() - t0) * 1000 / RUNS  # ms per run
    return elapsed

def bench_softmax(data, label):
    # Warmup
    for _ in range(WARMUP):
        out = mx.npx.softmax(data, axis=-1)
        out.wait_to_read()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        out = mx.npx.softmax(data, axis=-1)
        out.wait_to_read()
    elapsed = (time.perf_counter() - t0) * 1000 / RUNS
    return elapsed

omp = os.environ.get('OMP_NUM_THREADS', 'default')
print(f"\n=== B8 CPU Inference Benchmark ===")
print(f"OMP_NUM_THREADS={omp}  MXNet={mx.__version__}  ctx={CTX}")
print(f"DNNL_VERBOSE={os.environ.get('DNNL_VERBOSE', '0')}  MXNET_ONEDNN_DEBUG={os.environ.get('MXNET_ONEDNN_DEBUG', '0')}")
print()

results = []

# --- Conv2D 64ch (first-layer-style) ---
label = "Conv2D 64ch (1,3,224,224)"
net = nn.HybridSequential()
net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1))
data = mnp.random.uniform(0, 1, (1, 3, 224, 224))
data = mx.nd.array(data.asnumpy(), ctx=CTX)
data_np = mnp.array(data.asnumpy(), ctx=CTX)
net.hybridize()
ms = bench(net, data_np, label)
print(f"[{label}]  {ms:.3f} ms/inference")
results.append((label, omp, ms))

# --- Conv2D 512ch (mid-net style) ---
label = "Conv2D 512ch (1,3,224,224)"
net = nn.HybridSequential()
net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1))
data_np = mnp.ones((1, 3, 224, 224), ctx=CTX)
net.hybridize()
ms = bench(net, data_np, label)
print(f"[{label}]  {ms:.3f} ms/inference")
results.append((label, omp, ms))

# --- Dense 1000 ---
label = "Dense 1000 (1,2048)"
net = nn.HybridSequential()
net.add(nn.Dense(units=1000))
data_np = mnp.ones((1, 2048), ctx=CTX)
net.hybridize()
ms = bench(net, data_np, label)
print(f"[{label}]  {ms:.3f} ms/inference")
results.append((label, omp, ms))

# --- softmax ---
label = "softmax (1,1000)"
data_np = mnp.ones((1, 1000), ctx=CTX)
ms = bench_softmax(data_np, label)
print(f"[{label}]  {ms:.3f} ms/inference")
results.append((label, omp, ms))

print()
print("CSV: shape,omp,ms")
for r in results:
    print(f"  {r[0]},{r[1]},{r[2]:.3f}")
