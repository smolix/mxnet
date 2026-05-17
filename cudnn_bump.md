# cuDNN bump: 9.14.0 → 9.22.0

Issues.md item #17/#34 noted that cuDNN 9.0–9.3 ship heuristic tables that
do not cover sm_120 well: many conv shapes route to generic fallback engines
instead of Blackwell-tuned ones. We were on cuDNN 9.14.0 from the system
.deb (Ubuntu /usr/lib). The newest cuDNN for CUDA 13 is 9.22.0.52 (pip
`nvidia-cudnn-cu13`). This commit bumps to 9.22.0.

## What changed

1. **Local install of cuDNN 9.22 wheel** at
   `/workspace/mxnet/cudnn_local/unpacked/nvidia/cudnn/{include,lib}/`.
   `pip download --no-deps nvidia-cudnn-cu13==9.22.0.52` + `unzip`.
   The wheel's lib filenames are already the SONAMEs (`libcudnn.so.9`,
   not `libcudnn.so.9.22.0`), so no rename needed for bundling. We added
   unversioned `.so` symlinks (`libcudnn.so → libcudnn.so.9` etc) to make
   CMake's link step happy.

   The system /usr/lib cuDNN 9.14 is left untouched.

2. **CMake reconfigured** to point at the local include + libs:

   ```
   cmake -DCUDNN_INCLUDE=/workspace/mxnet/cudnn_local/unpacked/nvidia/cudnn/include \
         -DCUDNN_LIBRARY=/workspace/mxnet/cudnn_local/unpacked/nvidia/cudnn/lib/libcudnn.so \
         -DCUDNN_ROOT=/workspace/mxnet/cudnn_local/unpacked/nvidia/cudnn ..
   ```

   Verified by `ldd build/libmxnet.so | grep cudnn` →
   `/workspace/mxnet/cudnn_local/.../libcudnn.so.9`.

3. **Full rebuild** of mxnet target (~33 min). The cuDNN include path is
   tracked as a dep by CMake, so ~all object files re-compiled.

4. **`python/tools/bundle_runtime_libs.py` updated** to:
   - prefer the local-wheel lib dir over /usr/lib
   - drop the `.9.14.0` suffix from source filenames (the wheel uses the
     SONAME directly)
   - add `libcudnn_engines_tensor_ir.so.9` which is new in 9.22

5. **`bundle_runtime_libs.py` re-run**, packaging cuDNN 9.22 into
   `python/mxnet/lib/`. The wheel is now self-contained on 9.22.

## Perf results

3x3 conv 28x28 256→256 batch 32 FP32 with TF32 enabled (the TF32-audit
benchmark, `MXNET_CUDNN_AUTOTUNE_DEFAULT=2`):

| cuDNN     | per-iter | TFLOPS |
|-----------|---------:|-------:|
| 9.14.0    | 0.725 ms | 41.07  |
| **9.22.0** | **0.713 ms** | **41.52** |

Speedup ~1%. The audit reported 41.48 on 9.14 at the time; both fresh
runs are within run-to-run noise. This shape was already well-tuned at
9.14.

Sweep over five conv shapes (`bench_cudnn_sweep.py`) — TFLOPS:

| Shape                                    | 9.14   | 9.22   | Δ      |
|------------------------------------------|-------:|-------:|-------:|
| audit: 3x3 28x28 256→256 bs32            | 39.03  | 41.52  | +6%    |
| 1x1 14x14 512→2048 bs32                  | 29.58  | 29.93  | +1%    |
| 7x7 224x224 3→64 stride2 bs32 (mem-bound)|  7.10  |  7.24  | +2%    |
| dw 3x3 56x56 256→256 g=256 bs32          |  0.16  |  1.14  | **+7×** |
| gp 3x3 28x28 256→256 g=32 bs32           |  1.91  |  2.02  | +6%    |

The big win: **depthwise 3x3 jumps 7× (0.16 → 1.14 TFLOPS)**. This is
exactly the kind of sm_120-fallback case Issues.md flagged — on 9.14 the
heuristic picked a generic kernel that was nearly idle; 9.22 has a proper
sm_120 engine. Depthwise is still slow in absolute terms because it's
naturally bandwidth-bound on tiny per-channel kernels, but the floor is
lifted ~7×.

Grouped 3x3 and 1x1 expansion are essentially unchanged. The audit shape
gains a sliver. No shape regressed.

## Regression smoke tests

- `tests/python/dnnl/subgraphs/test_fc_subgraph.py`:
  **387 passed, 16 skipped** (exact match to the baseline 387/0/16).
- `tests/python/dnnl/test_dnnl.py`:
  **97 passed** (no failures; baseline was also clean).

Both ran with `--timeout=300` per the project memory.

## Verification

```
$ ldd python/mxnet/libmxnet.so | grep cudnn
        libcudnn.so.9 => python/mxnet/lib/libcudnn.so.9
$ python -c "import ctypes; for n in ['libcudnn_graph.so.9','libcudnn_cnn.so.9','libcudnn_ops.so.9','libcudnn.so.9']: ctypes.CDLL('python/mxnet/lib/'+n, mode=ctypes.RTLD_GLOBAL)
import ctypes; l=ctypes.CDLL('python/mxnet/lib/libcudnn.so.9'); l.cudnnGetVersion.restype=ctypes.c_size_t; print(l.cudnnGetVersion())"
92200
```

## Files changed

- `python/tools/bundle_runtime_libs.py` — wheel-bundling now pulls from
  `cudnn_local/unpacked/.../lib`, drops `.9.14.0` filename suffixes,
  adds `libcudnn_engines_tensor_ir.so.9` (new in 9.22).
- `bench_tf32_conv.py`, `bench_cudnn_sweep.py` — perf scripts (new).
- `cudnn_local/` — local wheel install (gitignored, large; not committed).
