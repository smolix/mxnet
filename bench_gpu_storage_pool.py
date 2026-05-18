#!/usr/bin/env python3
"""Benchmark MXNET_GPU_MEM_POOL_TYPE=Naive vs Round on GPU 0.

Runs ResNet-18 forward+backward at batch sizes [1, 8, 32, 128, 256] and
measures peak GPU memory allocated by MXNet.  Results are written to
storage_pool_bench.md.

Usage:
    python bench_gpu_storage_pool.py
"""

import os
import sys
import subprocess
import time
import gc

import numpy as np

# Must be done before importing mxnet in the subprocess — set env, then re-exec.
POOL_TYPE_ENV = "MXNET_GPU_MEM_POOL_TYPE"
CUDA_VISIBLE = "CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
BATCH_SIZES = [1, 8, 32, 128, 256]
POOL_TYPES = ["Naive", "Round"]


def _worker(pool_type: str, batch_size: int) -> dict:
    """Run one forward+backward pass and return peak memory info."""
    import mxnet as mx
    from mxnet import gluon
    from mxnet.gluon.model_zoo import vision as models

    ctx = mx.gpu(0)
    net = models.resnet18_v2(pretrained=False, classes=1000)
    net.initialize(mx.init.Xavier(), ctx=ctx)
    net.hybridize()

    dummy = mx.nd.random.uniform(0, 1, shape=(batch_size, 3, 224, 224), ctx=ctx)
    label = mx.nd.zeros((batch_size,), ctx=ctx)

    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    # Warmup pass
    with mx.autograd.record():
        out = net(dummy)
        loss = loss_fn(out, label)
    loss.backward()
    mx.nd.waitall()

    # Reset memory stats if available
    try:
        mx.nd.waitall()
        before_bytes = mx.context.gpu_memory_info(0)[1]  # free bytes
    except Exception:
        before_bytes = None

    # Timed pass
    t0 = time.perf_counter()
    with mx.autograd.record():
        out = net(dummy)
        loss = loss_fn(out, label)
    loss.backward()
    mx.nd.waitall()
    elapsed = time.perf_counter() - t0

    # Get GPU memory info
    try:
        total_bytes, free_bytes = mx.context.gpu_memory_info(0)
        peak_used_mb = (total_bytes - free_bytes) / (1024 ** 2)
        total_mb = total_bytes / (1024 ** 2)
    except Exception:
        peak_used_mb = -1
        total_mb = -1

    return {
        "pool_type": pool_type,
        "batch_size": batch_size,
        "peak_used_mb": peak_used_mb,
        "total_mb": total_mb,
        "elapsed_ms": elapsed * 1000,
        "oom": False,
    }


def run_one(pool_type: str, batch_size: int) -> dict:
    """Spawn a subprocess to isolate GPU memory state."""
    env = os.environ.copy()
    env[POOL_TYPE_ENV] = pool_type
    env[CUDA_VISIBLE_DEVICES] = "0"
    # Pass batch size and pool type via env
    env["_BENCH_POOL_TYPE"] = pool_type
    env["_BENCH_BATCH_SIZE"] = str(batch_size)

    script = """
import os, sys, json
sys.path.insert(0, '/workspace/mxnet/python')
pool_type = os.environ['_BENCH_POOL_TYPE']
batch_size = int(os.environ['_BENCH_BATCH_SIZE'])

import mxnet as mx
import mxnet.numpy as mnp
from mxnet import gluon, npx
from mxnet.gluon.model_zoo import vision as models
import time

npx.set_np()
dev = mx.gpu(0)
net = models.resnet18_v2(pretrained=False, classes=1000)
net.initialize(mx.init.Xavier(), ctx=dev)
net.hybridize()

dummy = mnp.random.uniform(0, 1, size=(batch_size, 3, 224, 224)).as_in_ctx(dev)
label = mnp.zeros((batch_size,)).as_in_ctx(dev)
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

# Warmup
with mx.autograd.record():
    out = net(dummy)
    loss = loss_fn(out, label)
loss.backward()
mx.nd.waitall()

# Timed
t0 = time.perf_counter()
with mx.autograd.record():
    out = net(dummy)
    loss = loss_fn(out, label)
loss.backward()
mx.nd.waitall()
elapsed = time.perf_counter() - t0

try:
    free_bytes, total_bytes = mx.context.gpu_memory_info(0)
    peak_used_mb = (total_bytes - free_bytes) / (1024 ** 2)
    total_mb = total_bytes / (1024 ** 2)
except Exception:
    peak_used_mb = -1
    total_mb = -1

result = {
    'pool_type': pool_type,
    'batch_size': batch_size,
    'peak_used_mb': peak_used_mb,
    'total_mb': total_mb,
    'elapsed_ms': elapsed * 1000,
    'oom': False,
}
print('RESULT:' + json.dumps(result))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr[-500:]
            if "out of memory" in stderr.lower() or "OOM" in stderr:
                return {
                    "pool_type": pool_type,
                    "batch_size": batch_size,
                    "peak_used_mb": None,
                    "total_mb": None,
                    "elapsed_ms": None,
                    "oom": True,
                }
            print(f"  STDERR: {stderr[-300:]}", file=sys.stderr)
            return {
                "pool_type": pool_type,
                "batch_size": batch_size,
                "peak_used_mb": None,
                "total_mb": None,
                "elapsed_ms": None,
                "oom": False,
                "error": stderr[-200:],
            }
        # Parse result
        for line in result.stdout.splitlines():
            if line.startswith("RESULT:"):
                import json
                return json.loads(line[len("RESULT:"):])
        return {
            "pool_type": pool_type,
            "batch_size": batch_size,
            "peak_used_mb": None,
            "total_mb": None,
            "elapsed_ms": None,
            "oom": False,
            "error": "no RESULT line",
        }
    except subprocess.TimeoutExpired:
        return {
            "pool_type": pool_type,
            "batch_size": batch_size,
            "peak_used_mb": None,
            "total_mb": None,
            "elapsed_ms": None,
            "oom": False,
            "error": "timeout",
        }


def main():
    results = []
    print(f"{'Pool':<8} {'Batch':>6}  {'Peak (MiB)':>12}  {'Time (ms)':>10}  {'Status'}")
    print("-" * 60)
    for pool_type in POOL_TYPES:
        for bs in BATCH_SIZES:
            print(f"  Running {pool_type:6s} bs={bs:4d} ... ", end="", flush=True)
            r = run_one(pool_type, bs)
            results.append(r)
            if r.get("oom"):
                print("OOM")
            elif r.get("error"):
                print(f"ERROR: {r['error'][:60]}")
            else:
                print(f"peak={r['peak_used_mb']:.0f} MiB  t={r['elapsed_ms']:.1f} ms")

    _write_markdown(results)
    print("\nResults written to storage_pool_bench.md")
    return results


def _write_markdown(results):
    import json

    # Organize by pool type
    naive = {r["batch_size"]: r for r in results if r["pool_type"] == "Naive"}
    round_ = {r["batch_size"]: r for r in results if r["pool_type"] == "Round"}

    lines = [
        "# GPU Storage Pool Benchmark: Naive vs Round",
        "",
        "**Date:** 2026-05-18  ",
        "**GPU:** NVIDIA RTX PRO 4000 Blackwell (sm_120, 24 GiB)  ",
        "**Model:** ResNet-18 v2 (Gluon model zoo, random weights)  ",
        "**Task:** forward + backward, 1 warmup pass + 1 timed pass  ",
        "**Method:** isolated subprocess per (pool_type, batch_size); "
        "`mx.context.gpu_memory_info(0)` after `waitall()`  ",
        "",
        "## Results",
        "",
        "| Batch | Naive peak (MiB) | Round peak (MiB) | Naive time (ms) | Round time (ms) | Delta peak | Winner |",
        "|------:|----------------:|----------------:|---------------:|---------------:|:----------:|:------:|",
    ]

    max_naive_ok = None
    max_round_ok = None

    for bs in BATCH_SIZES:
        n = naive.get(bs, {})
        r = round_.get(bs, {})

        def fmt_mem(d):
            if d.get("oom"):
                return "OOM"
            if d.get("error"):
                return "ERR"
            v = d.get("peak_used_mb")
            return f"{v:.0f}" if v is not None else "?"

        def fmt_time(d):
            if d.get("oom") or d.get("error"):
                return "—"
            v = d.get("elapsed_ms")
            return f"{v:.1f}" if v is not None else "?"

        nm = fmt_mem(n)
        rm = fmt_mem(r)
        nt = fmt_time(n)
        rt = fmt_time(r)

        # Compute delta
        n_val = n.get("peak_used_mb") if not n.get("oom") and not n.get("error") else None
        r_val = r.get("peak_used_mb") if not r.get("oom") and not r.get("error") else None

        if n_val is not None and r_val is not None:
            delta_pct = (r_val - n_val) / n_val * 100
            delta_str = f"{delta_pct:+.1f}%"
            winner = "Round" if delta_pct < -5 else ("Naive" if delta_pct > 5 else "tie")
            max_naive_ok = bs
            max_round_ok = bs
        elif n.get("oom") and not r.get("oom") and r_val is not None:
            delta_str = "Naive OOM"
            winner = "Round"
            max_round_ok = bs
        elif r.get("oom") and not n.get("oom") and n_val is not None:
            delta_str = "Round OOM"
            winner = "Naive"
            max_naive_ok = bs
        else:
            delta_str = "—"
            winner = "—"

        lines.append(
            f"| {bs:5d} | {nm:>16} | {rm:>16} | {nt:>15} | {rt:>15} | {delta_str:>10} | {winner:>6} |"
        )

    # Analysis
    lines += [
        "",
        "## Analysis",
        "",
    ]

    # Compute overall verdict
    deltas = []
    for bs in BATCH_SIZES:
        n = naive.get(bs, {})
        r = round_.get(bs, {})
        n_val = n.get("peak_used_mb") if not n.get("oom") and not n.get("error") else None
        r_val = r.get("peak_used_mb") if not r.get("oom") and not r.get("error") else None
        if n_val and r_val:
            deltas.append((r_val - n_val) / n_val * 100)

    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        min_delta = min(deltas)
        max_delta = max(deltas)
        lines += [
            f"- Across {len(deltas)} comparable data points, Round uses on average "
            f"**{avg_delta:+.1f}%** memory vs Naive (negative = Round uses less).",
            f"- Range: {min_delta:+.1f}% to {max_delta:+.1f}%.",
        ]
        if avg_delta < -20:
            lines += [
                "",
                "**Verdict: `MXNET_GPU_MEM_POOL_TYPE=Round` is clearly better** — average "
                f"{-avg_delta:.0f}% less peak GPU memory. Recommend documenting this as the "
                "preferred setting for 24 GiB cards.",
            ]
        elif avg_delta > 5:
            lines += [
                "",
                "**Verdict: Naive is slightly better** — Round uses more peak memory on average. "
                "Likely because Round pre-rounds up allocations to the next power-of-two, "
                "increasing waste on this model/batch-size combination.",
            ]
        else:
            lines += [
                "",
                f"**Verdict: no significant difference** (avg delta {avg_delta:+.1f}%, within ±5%). "
                "Changing `MXNET_GPU_MEM_POOL_TYPE` is not a free win for ResNet-18 on this GPU. "
                "The choice may matter more for models with irregular allocation patterns; "
                "benchmark on your specific workload if fragmentation is observed.",
            ]
    else:
        lines += [
            "- Not enough comparable data points to draw a conclusion.",
        ]

    lines += [
        "",
        "## Raw JSON",
        "",
        "```json",
        json.dumps(results, indent=2),
        "```",
    ]

    with open("storage_pool_bench.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
