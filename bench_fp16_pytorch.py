#!/usr/bin/env python3
"""PyTorch-only fp16 benchmark. Outputs JSON to /workspace/mxnet/fp16_bench_pytorch.json"""
import os, time, json, statistics
import torch
import torch.nn as nn

WARMUP = 5
RUNS = 20
GPU_ID = int(os.environ.get("BENCH_GPU", "0"))
device = torch.device(f"cuda:{GPU_ID}")

def sync():
    torch.cuda.synchronize(device)

def time_fn(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times), statistics.stdev(times)

results = []

def record(kernel, shape_str, ms, std, notes=""):
    results.append(dict(kernel=kernel, shape=shape_str, ms=ms, std=std, notes=notes))
    print(f"  {kernel:12s} {shape_str:35s}  {ms:8.3f} ms  (±{std:.3f})")

# 1. Conv2D
print("=== Conv2D ===")
for batch in [32, 64, 128]:
    conv = nn.Conv2d(256, 256, kernel_size=3, padding=1).half().to(device)
    x = torch.randn(batch, 256, 32, 32, dtype=torch.float16, device=device)
    sync()
    ms, std = time_fn(lambda: conv(x))
    record("Conv2D", f"batch={batch},C=256,HW=32", ms, std)

# 2. Dense/matmul
print("=== Dense ===")
for M, N, K in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]:
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)
    sync()
    ms, std = time_fn(lambda: torch.mm(A, B))
    record("Dense", f"MNK=({M},{N},{K})", ms, std)

# 3. Softmax
print("=== Softmax ===")
softmax = nn.Softmax(dim=-1)
for batch, hidden in [(32, 4096), (128, 16384)]:
    x = torch.randn(batch, hidden, dtype=torch.float16, device=device)
    sync()
    ms, std = time_fn(lambda: softmax(x))
    record("Softmax", f"({batch},{hidden})", ms, std)

# 4. LayerNorm
print("=== LayerNorm ===")
for batch, seq, hidden in [(32, 128, 768), (8, 1024, 4096)]:
    ln = nn.LayerNorm(hidden).half().to(device)
    x = torch.randn(batch, seq, hidden, dtype=torch.float16, device=device)
    sync()
    ms, std = time_fn(lambda: ln(x))
    record("LayerNorm", f"({batch},{seq},{hidden})", ms, std)

# 5. Add
print("=== Add ===")
for numel in [1_000_000, 4_000_000, 16_000_000]:
    a = torch.randn(numel, dtype=torch.float16, device=device)
    b = torch.randn(numel, dtype=torch.float16, device=device)
    sync()
    ms, std = time_fn(lambda: a + b)
    record("Add", f"{numel//1_000_000}M", ms, std)

# 6. Mul
print("=== Mul ===")
for numel in [1_000_000, 4_000_000, 16_000_000]:
    a = torch.randn(numel, dtype=torch.float16, device=device)
    b = torch.randn(numel, dtype=torch.float16, device=device)
    sync()
    ms, std = time_fn(lambda: a * b)
    record("Mul", f"{numel//1_000_000}M", ms, std)

with open("/workspace/mxnet/fp16_bench_pytorch.json", "w") as f:
    json.dump({"framework": "pytorch", "version": torch.__version__,
               "device": torch.cuda.get_device_name(GPU_ID),
               "sm_cap": list(torch.cuda.get_device_capability(GPU_ID)),
               "warmup": WARMUP, "runs": RUNS, "results": results}, f, indent=2)
print("Saved /workspace/mxnet/fp16_bench_pytorch.json")
