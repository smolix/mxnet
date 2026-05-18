# GPU Storage Pool Benchmark: Naive vs Round

**Date:** 2026-05-18  
**GPU:** NVIDIA RTX PRO 4000 Blackwell (sm_120, 24 GiB)  
**Model:** ResNet-18 v2 (Gluon model zoo, random weights)  
**Task:** forward + backward, 1 warmup pass + 1 timed pass  
**Method:** isolated subprocess per (pool_type, batch_size); `mx.context.gpu_memory_info(0)` after `waitall()`  

## Results

| Batch | Naive peak (MiB) | Round peak (MiB) | Naive time (ms) | Round time (ms) | Delta peak | Winner |
|------:|----------------:|----------------:|---------------:|---------------:|:----------:|:------:|
|     1 |             4427 |             4485 |             9.1 |             8.2 |      +1.3% |    tie |
|     8 |             4721 |             4793 |            10.8 |            12.4 |      +1.5% |    tie |
|    32 |             5591 |             5825 |            26.2 |            26.0 |      +4.2% |    tie |
|   128 |             8955 |             9487 |           105.8 |           103.8 |      +5.9% |  Naive |
|   256 |            13459 |            14081 |           214.3 |           215.6 |      +4.6% |    tie |

## Analysis

- Across 5 comparable data points, Round uses on average **+3.5%** memory vs Naive (negative = Round uses less).
- Range: +1.3% to +5.9%.

**Verdict: no significant difference** (avg delta +3.5%, within ±5%). Changing `MXNET_GPU_MEM_POOL_TYPE` is not a free win for ResNet-18 on this GPU. The choice may matter more for models with irregular allocation patterns; benchmark on your specific workload if fragmentation is observed.

## Raw JSON

```json
[
  {
    "pool_type": "Naive",
    "batch_size": 1,
    "peak_used_mb": 4426.875,
    "total_mb": 23986.0,
    "elapsed_ms": 9.109951090067625,
    "oom": false
  },
  {
    "pool_type": "Naive",
    "batch_size": 8,
    "peak_used_mb": 4720.875,
    "total_mb": 23986.0,
    "elapsed_ms": 10.83139842376113,
    "oom": false
  },
  {
    "pool_type": "Naive",
    "batch_size": 32,
    "peak_used_mb": 5590.875,
    "total_mb": 23986.0,
    "elapsed_ms": 26.18552977219224,
    "oom": false
  },
  {
    "pool_type": "Naive",
    "batch_size": 128,
    "peak_used_mb": 8954.875,
    "total_mb": 23986.0,
    "elapsed_ms": 105.82157364115119,
    "oom": false
  },
  {
    "pool_type": "Naive",
    "batch_size": 256,
    "peak_used_mb": 13458.875,
    "total_mb": 23986.0,
    "elapsed_ms": 214.27844371646643,
    "oom": false
  },
  {
    "pool_type": "Round",
    "batch_size": 1,
    "peak_used_mb": 4484.875,
    "total_mb": 23986.0,
    "elapsed_ms": 8.15140875056386,
    "oom": false
  },
  {
    "pool_type": "Round",
    "batch_size": 8,
    "peak_used_mb": 4792.875,
    "total_mb": 23986.0,
    "elapsed_ms": 12.39461312070489,
    "oom": false
  },
  {
    "pool_type": "Round",
    "batch_size": 32,
    "peak_used_mb": 5824.875,
    "total_mb": 23986.0,
    "elapsed_ms": 26.033313013613224,
    "oom": false
  },
  {
    "pool_type": "Round",
    "batch_size": 128,
    "peak_used_mb": 9486.875,
    "total_mb": 23986.0,
    "elapsed_ms": 103.76100614666939,
    "oom": false
  },
  {
    "pool_type": "Round",
    "batch_size": 256,
    "peak_used_mb": 14080.875,
    "total_mb": 23986.0,
    "elapsed_ms": 215.55314166471362,
    "oom": false
  }
]
```
