"""Performance scan for MXNet on the local GPU (factual baseline).

Measures matmul / conv / elementwise / reduction / softmax / dispatch and
reports throughput against RTX 4090 (Ada, sm_89) theoretical peaks:
  fp16 tensor (fp32 accum) ~165 TFLOP/s, TF32 ~82, fp32 ~82.6 TFLOP/s,
  HBM/GDDR6X bandwidth ~1008 GB/s.
Run with the mxnet env vars set (MXNET_LIBRARY_PATH, PYTHONPATH, LD_PRELOAD).
"""
import time
import mxnet as mx
from mxnet import nd

CTX = mx.gpu(0)
# RTX 4090 (Ada) approximate peaks. fp32 matmul is reported against TF32 peak
# because MXNet enables TF32 tensor cores for fp32 GEMM by default.
PEAK = {"float32": 82.6, "float16": 165.0, "bfloat16": 165.0}
BW_PEAK = 1008.0  # GB/s (GDDR6X)


def sync():
    mx.nd.waitall()


def timed(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(iters):
        out = fn()
    if isinstance(out, nd.NDArray):
        out.wait_to_read()
    sync()
    return (time.time() - t0) / iters


def matmul():
    print("\n== MATMUL (TFLOP/s, % of peak) ==")
    # fp16/fp32 via nd.dot; bf16 GEMM is reached via FullyConnected (nd.dot
    # intentionally rejects bf16), so measure it that way for an apples count.
    for dt in ["float32", "float16"]:
        for N in [1024, 2048, 4096]:
            a = nd.random.uniform(-1, 1, (N, N), ctx=CTX).astype(dt)
            b = nd.random.uniform(-1, 1, (N, N), ctx=CTX).astype(dt)
            dt_s = timed(lambda: nd.dot(a, b), 50)
            tflops = 2.0 * N ** 3 / dt_s / 1e12
            peak = PEAK[dt]
            print(f"  dot {dt:8} {N:5}: {dt_s*1e3:7.2f} ms  {tflops:7.1f} TFLOP/s  ({tflops/peak*100:4.0f}% of {peak})")
    for N in [2048, 4096]:
        try:
            x = nd.random.uniform(-1, 1, (N, N), ctx=CTX).astype('bfloat16')
            w = nd.random.uniform(-1, 1, (N, N), ctx=CTX).astype('bfloat16')
            dt_s = timed(lambda: nd.FullyConnected(x, w, no_bias=True, num_hidden=N), 50)
            tflops = 2.0 * N ** 3 / dt_s / 1e12
            print(f"  FC  bfloat16 {N:5}: {dt_s*1e3:7.2f} ms  {tflops:7.1f} TFLOP/s  ({tflops/PEAK['bfloat16']*100:4.0f}% of {PEAK['bfloat16']})")
        except Exception as e:
            print(f"  FC  bfloat16 {N:5}: ERROR {str(e).splitlines()[-1][:50]}")


def conv():
    print("\n== CONV2D fwd (cuDNN) ==")
    for (n, c, h, oc, k) in [(32, 64, 56, 64, 3), (32, 256, 28, 256, 3)]:
        try:
            x = nd.random.uniform(shape=(n, c, h, h), ctx=CTX)
            w = nd.random.uniform(shape=(oc, c, k, k), ctx=CTX)
            dt_s = timed(lambda: nd.Convolution(x, w, no_bias=True, kernel=(k, k),
                                                num_filter=oc, pad=(1, 1)), 30)
            out_h = h
            flops = 2.0 * n * oc * out_h * out_h * c * k * k
            print(f"  N{n} C{c}->{oc} {h}x{h} k{k}: {dt_s*1e3:7.2f} ms  {flops/dt_s/1e12:6.1f} TFLOP/s")
        except Exception as e:
            print(f"  conv {c}->{oc}: ERROR {str(e).splitlines()[-1][:50]}")


def timed_sync(fn, iters, warmup=10):
    """Time fn syncing every iteration (accurate per-op wall time)."""
    for _ in range(warmup):
        fn().wait_to_read()
    sync()
    t0 = time.time()
    for _ in range(iters):
        fn().wait_to_read()
    sync()
    return (time.time() - t0) / iters


def bandwidth():
    print("\n== MEMORY-BOUND ops (GB/s, % of ~1008 peak; synced per-op) ==")
    N = 1 << 24  # 16M elements
    for dt, bytes_per in [("float32", 4), ("float16", 2)]:
        a = nd.random.uniform(-1, 1, (N,), ctx=CTX).astype(dt)
        # elementwise add: read 2, write 1
        dt_s = timed_sync(lambda: a + a, 200)
        gb = 3 * N * bytes_per / dt_s / 1e9
        print(f"  add        {dt:8}: {dt_s*1e3:6.3f} ms  {gb:6.0f} GB/s ({gb/BW_PEAK*100:3.0f}%)")
        # reduction sum: read 1
        dt_s = timed_sync(lambda: nd.sum(a), 200)
        gb = N * bytes_per / dt_s / 1e9
        print(f"  sum        {dt:8}: {dt_s*1e3:6.3f} ms  {gb:6.0f} GB/s ({gb/BW_PEAK*100:3.0f}%)")


def softmax():
    print("\n== softmax / layernorm (common NN ops) ==")
    x = nd.random.uniform(-1, 1, (4096, 4096), ctx=CTX)
    print(f"  softmax 4096x4096: {timed(lambda: nd.softmax(x), 100)*1e3:6.3f} ms")
    g = nd.ones((4096,), ctx=CTX); b = nd.zeros((4096,), ctx=CTX)
    print(f"  layernorm 4096x4096: {timed(lambda: nd.LayerNorm(x, g, b), 100)*1e3:6.3f} ms")


def dispatch():
    print("\n== DISPATCH overhead (tiny ops) ==")
    x = nd.ones((32, 32), ctx=CTX)
    # latency of a single tiny op (sync each)
    def one():
        y = x + 1; y.wait_to_read(); return y
    dt_s = timed(one, 2000)
    print(f"  tiny add (sync each):   {dt_s*1e6:6.1f} us/op")
    # async throughput (no per-op sync)
    dt_s = timed(lambda: x + 1, 5000, warmup=50)
    print(f"  tiny add (async queue): {dt_s*1e6:6.1f} us/op")


if __name__ == "__main__":
    print(f"GPU: {mx.runtime.feature_list and 'mxnet'} ctx={CTX}")
    matmul()
    conv()
    bandwidth()
    softmax()
    dispatch()
    print("\nscan done.")
