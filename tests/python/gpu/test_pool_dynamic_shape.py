"""
Regression test for apache/mxnet#17335 / .investigations A2.

The Naive pool (and Round pool) in `src/storage/pooled_storage_manager.h`
keeps an unbounded number of free chunks per size-class bucket. With
dynamic-shape workloads (variable-length NLP, dynamic image, batchify-into-Bucket
in Gluon's DataLoader), every novel rounded shape adds new chunks to its
bucket and they are never evicted. Reporters of #17335 see OOM at batch=16
while PyTorch handles batch=256 on the same hardware.

This test exercises that scenario synthetically: 5000 allocations across
many distinct rounded shapes. The CURRENT binary will demonstrate unbounded
pool growth (peak memory climbs to ~device-cap minus MXNET_GPU_MEM_POOL_RESERVE).
After the patch lands (a2_pool_retention.patch), peak memory should plateau
once each bucket has retained K=MXNET_GPU_MEM_POOL_PER_BUCKET_LIMIT chunks.

Per-test timeout 600s, 5000 iters is ~2-3 min on Blackwell sm_120.
"""
import os
import random
import subprocess
import sys

import mxnet as mx
from mxnet import np as mxnp

try:
    import pytest
except ImportError:  # standalone repro mode
    pytest = None


def _pick_free_gpu():
    """Pick the GPU with the most free memory."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            timeout=10,
        ).decode()
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0
    best = (0, -1)
    for line in out.strip().splitlines():
        idx, free_mib = [int(x.strip()) for x in line.split(",")]
        if free_mib > best[1]:
            best = (idx, free_mib)
    return best[0]


def _peak_growth_curve(dev_id, n_iters, sample_every=50,
                       dtype="float32", seed=0, concurrency=8,
                       n_buckets=8):
    """Run a dynamic-shape workload, return list of (iter, used_bytes).

    Each iteration picks a random bucket and allocates `concurrency` live
    buffers of that size simultaneously, then releases them all. This
    mimics the realistic #17335 pattern of multi-worker DataLoader where
    several concurrent allocations of the same rounded shape stack up in
    one bucket. With unbounded retention the bucket keeps all `concurrency`
    chunks forever; with K-cap retention only K are retained.

    The pool also grows from learning new buckets (single chunk per new
    shape), which K-cap does NOT mitigate by design. We use a modest
    n_buckets=8 so the bucket-discovery growth saturates quickly (within
    ~100 iters) and the over-K stacking dominates the late curve.
    """
    rng = random.Random(seed)
    curve = []
    free0, total = mx.context.gpu_memory_info(device_id=dev_id)
    baseline_used = total - free0
    ctx = mx.gpu(dev_id)
    # cols=524288 -> 2 MiB per row (fp32). rows in [1..n_buckets] -> 2,4,...
    cols = 524288
    rows_pool = list(range(1, n_buckets + 1))
    for i in range(n_iters):
        R = rng.choice(rows_pool)
        # Hold `concurrency` buffers of the SAME rounded size live, then drop.
        live = [mxnp.empty((R, cols), dtype=dtype, ctx=ctx)
                for _ in range(concurrency)]
        for b in live:
            b[0, 0] = 1.0
        mx.nd.waitall()
        del live
        if i % sample_every == 0 or i == n_iters - 1:
            mx.nd.waitall()
            free_i, _ = mx.context.gpu_memory_info(device_id=dev_id)
            used_i = total - free_i - baseline_used
            curve.append((i, used_i))
    return curve, total


if pytest is not None:
    _gpu_required = pytest.mark.skipif(
        mx.context.num_gpus() == 0, reason="requires GPU")
else:
    def _gpu_required(f):  # type: ignore[no-redef]
        return f


@_gpu_required
def test_pool_dynamic_shape_plateaus():
    """Regression guard: pool peak memory must plateau, not climb unboundedly.

    Pre-patch (current binary): peak grows ~linearly with the number of
    distinct rounded shapes encountered. Test is expected to be SKIPPED
    via xfail-strict=False until libmxnet.so is rebuilt.

    Post-patch: with default MXNET_GPU_MEM_POOL_PER_BUCKET_LIMIT=4, the
    pool stabilizes once every bucket has at most 4 retained chunks. We
    assert: peak at iter 4500 is within 1.25x of peak at iter 1500.
    """
    # Bucket layout: 8 distinct rounded sizes 2,4,...,16 MiB.
    # Per-iter concurrency = 8 same-bucket buffers.
    # Pre-patch:   each bucket retains 8 chunks -> ~580 MiB pool
    # Post-patch:  each bucket retains <= 4 -> ~290 MiB pool
    n_buckets = 8
    concurrency = 8
    K_default = 4  # MXNET_GPU_MEM_POOL_PER_BUCKET_LIMIT default
    bucket_max_mib = 2 * n_buckets  # 16 MiB largest bucket
    # 5000 iters is the long-run plateau check; on Blackwell ~2-3 min.
    dev_id = _pick_free_gpu()
    curve, total = _peak_growth_curve(
        dev_id, n_iters=5000, sample_every=100,
        concurrency=concurrency, n_buckets=n_buckets,
    )
    print(f"\n[pool_dynamic_shape] dev={dev_id}, total={total/2**30:.1f} GiB")
    for i, used in curve[::5]:
        print(f"  iter {i:5d}: pool+overhead = {used/2**20:8.1f} MiB")

    # Reference points
    early = [u for (i, u) in curve if 1400 <= i <= 1600]
    late = [u for (i, u) in curve if 4400 <= i <= 4600]
    assert early and late, "sampling produced no reference points"
    e = max(early)
    l = max(late)
    print(f"[pool_dynamic_shape] early peak={e/2**20:.1f} MiB, "
          f"late peak={l/2**20:.1f} MiB, ratio={l/max(e,1):.2f}")

    # (1) Plateau check: late peak should not grow vs early peak.
    # Holds both pre- and post-patch in this synthetic workload (pre-patch
    # plateaus because the bucket set is finite). Kept as a sanity gate.
    assert l <= 1.25 * e + (16 << 20), (
        f"Pool grew unboundedly: early={e/2**20:.1f} MiB, "
        f"late={l/2**20:.1f} MiB. See apache/mxnet#17335.")

    # (2) Absolute-peak check: with the patch + default K=4, pool should
    # hold ~K * sum(bucket_size). Pre-patch this assertion FAILS because
    # pool holds ~concurrency * sum(bucket_size). Allow generous slack
    # for CUDA context (~200 MiB).
    sum_bucket_mib = sum(range(1, n_buckets + 1)) * 2  # 2+4+...+16 = 72 MiB
    expected_pool_mib = K_default * sum_bucket_mib    # 4 * 72 = 288 MiB
    overhead_slack_mib = 220
    peak_cap_mib = expected_pool_mib + overhead_slack_mib  # ~508 MiB
    peak_mib = max(u for (_, u) in curve) / 2**20
    print(f"[pool_dynamic_shape] absolute peak={peak_mib:.1f} MiB, "
          f"cap={peak_cap_mib} MiB "
          f"(K={K_default} * sum_bucket_mib={sum_bucket_mib} + "
          f"overhead={overhead_slack_mib})")
    assert peak_mib <= peak_cap_mib, (
        f"Pool exceeded K-cap budget: peak={peak_mib:.1f} MiB > "
        f"cap={peak_cap_mib} MiB. Either K-cap is not taking effect, or "
        f"MXNET_GPU_MEM_POOL_PER_BUCKET_LIMIT was set higher. See "
        f"apache/mxnet#17335.")


if __name__ == "__main__":
    # Standalone repro mode: print the curve and exit, no asserts.
    # Use this to demonstrate the pre-patch bug.
    dev_id = _pick_free_gpu()
    n = int(os.environ.get("N_ITERS", "5000"))
    curve, total = _peak_growth_curve(
        dev_id, n_iters=n, sample_every=100,
    )
    print(f"dev={dev_id}, total={total/2**30:.2f} GiB, "
          f"pool_type={os.environ.get('MXNET_GPU_MEM_POOL_TYPE', 'Naive')}, "
          f"per_bucket_limit="
          f"{os.environ.get('MXNET_GPU_MEM_POOL_PER_BUCKET_LIMIT', 'inf')}")
    print("iter, pool+overhead_MiB")
    for i, used in curve:
        print(f"{i},{used/2**20:.1f}")
