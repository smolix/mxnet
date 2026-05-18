#!/usr/bin/env python3
"""
fp16 benchmark: MXNet vs PyTorch on NVIDIA RTX PRO 4000 Blackwell (sm_120)

Because MXNet (numpy 1.x) and PyTorch (numpy 2.x) are installed in separate
venvs with incompatible numpy versions, this script runs each framework in
its own subprocess, collects JSON results, then prints the merged comparison table.

Usage (recommended — autotune enabled for MXNet):
  MXNET_CUDNN_AUTOTUNE_DEFAULT=1 CUDA_VISIBLE_DEVICES=0 python bench_fp16_mxnet_vs_pytorch.py

Individual sub-scripts can also be run directly:
  CUDA_VISIBLE_DEVICES=0 /workspace/mxnet/.venv/bin/python             bench_fp16_mxnet.py
  CUDA_VISIBLE_DEVICES=0 /workspace/d2l-neu/.venv-pytorch/bin/python   bench_fp16_pytorch.py

Kernels benchmarked (each: 5 warmup + 20 timed runs, fp16):
  Conv2D       3x3, in/out=256, HW=32, batch in {32, 64, 128}
  Dense        mm MNK in {(1024^3), (4096^3), (8192^3)}
  Softmax      over last axis, shapes (32,4096) and (128,16384)
  LayerNorm    shapes (32,128,768) and (8,1024,4096)
  Add / Mul    elementwise fp16 at 1M, 4M, 16M elements
"""

import json
import os
import subprocess
import sys

MXNET_PYTHON  = "/workspace/mxnet/.venv/bin/python"
PYTORCH_PYTHON = "/workspace/d2l-neu/.venv-pytorch/bin/python"
MXNET_SCRIPT   = "/workspace/mxnet/bench_fp16_mxnet.py"
PYTORCH_SCRIPT = "/workspace/mxnet/bench_fp16_pytorch.py"
MX_JSON  = "/workspace/mxnet/fp16_bench_mxnet.json"
PT_JSON  = "/workspace/mxnet/fp16_bench_pytorch.json"

def run_bench(python, script, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"\n{'='*60}")
    print(f"Running: {python} {script}")
    print('='*60)
    result = subprocess.run([python, script], env=env, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: subprocess exited with code {result.returncode}", file=sys.stderr)

def load(path):
    with open(path) as f:
        return json.load(f)

def print_table(mx_data, pt_data):
    mx_map = {(r["kernel"], r["shape"]): r for r in mx_data["results"]}
    pt_map = {(r["kernel"], r["shape"]): r for r in pt_data["results"]}

    header = f"{'Kernel':<12} {'Shape':<37} {'MX fp16 ms':>12} {'PT fp16 ms':>12} {'Speedup MX/PT':>14}  Notes"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    flags = []
    for key in mx_map:
        mx_r = mx_map[key]
        pt_r = pt_map.get(key)
        if pt_r is None:
            continue
        mx_ms = mx_r["ms"]
        pt_ms = pt_r["ms"]
        speedup = pt_ms / mx_ms if mx_ms > 0 else float("nan")
        note = ""
        if speedup < 0.5:
            note = "<-- MXNet >2x slower"
            flags.append((key, speedup))
        elif speedup > 2.0:
            note = "<-- MXNet >2x faster"
        print(f"{key[0]:<12} {key[1]:<37} {mx_ms:>12.3f} {pt_ms:>12.3f} {speedup:>14.3f}  {note}")
    print(sep)
    print(f"\nMXNet {mx_data['version']}  vs  PyTorch {pt_data['version']}")
    print(f"Device: {pt_data.get('device','unknown')}  SM cap: {pt_data.get('sm_cap','?')}")
    if flags:
        print(f"\nActionable gaps (MXNet >2x slower than PyTorch):")
        for k, s in flags:
            print(f"  {k[0]:12s} {k[1]:35s}  speedup={s:.3f}")
    else:
        print("\nNo kernels where MXNet is >2x slower than PyTorch.")

if __name__ == "__main__":
    skip_run = "--no-run" in sys.argv  # use pre-existing JSON files
    if not skip_run:
        run_bench(MXNET_PYTHON, MXNET_SCRIPT,
                  extra_env={"MXNET_CUDNN_AUTOTUNE_DEFAULT": "1"})
        run_bench(PYTORCH_PYTHON, PYTORCH_SCRIPT)

    mx_data = load(MX_JSON)
    pt_data = load(PT_JSON)
    print_table(mx_data, pt_data)
