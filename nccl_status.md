# NCCL Status — smolix/mxnet Blackwell port

Hardware: 2× NVIDIA RTX PRO 4000 Blackwell (sm_120), PCIe 4.0 x16
NCCL: 2.28+ (bundled in wheel)
Build: `-DUSE_NCCL=1`
Test date: 2026-05-18

---

## Single-process / multi-GPU (supported)

`kv.create('nccl')` creates a `KVStoreNCCL` instance (extends `KVStoreLocal`,
calls `ncclCommInitAll()` to span all visible GPUs in one process).

### Correctness results — 10/10 PASS

| Test | Result |
|---|---|
| float32 small (1024-element / 4 KiB) | PASS |
| float32 large (1M-element / 4 MiB) | PASS |
| float16 push/pull | PASS |
| uint8 push/pull | PASS |
| multi-dimensional shape (4, 32, 32) | PASS |
| multiple keys (4 keys, init-all-before-push) | PASS |
| bandwidth 1 MiB | PASS |
| bandwidth 16 MiB | PASS |
| bandwidth 256 MiB | PASS |
| int8 NOT supported (expect error) | PASS |

Run: `pytest tests/python/gpu/test_nccl_singleproc.py -v --timeout=300`

### Bandwidth (push + pull round-trip, float32)

| Tensor size | GB/s | Notes |
|---|---|---|
| 1 MiB | ~9 GB/s | Small-message overhead dominates |
| 16 MiB | ~19–20 GB/s | ~60–65% of PCIe 4.0 x16 peak |
| 256 MiB | ~20–22 GB/s | ~65–70% of PCIe 4.0 x16 peak |

PCIe 4.0 x16 theoretical peak: 32 GB/s (unidirectional), 64 GB/s (bidirectional
full duplex). Push+pull together is bidirectional; measured ~20–22 GB/s means
~30–35% of full-duplex or ~65% of one-directional capacity. This is normal for
NCCL all-reduce over PCIe — the ring all-reduce algorithm sends each element
multiple times. For NVLink hosts the numbers would be 5–10× higher.

### Dtype support

| dtype | Supported |
|---|---|
| float32 | Yes |
| float16 | Yes |
| float64 | Yes (not tested; in GetNCCLType switch) |
| uint8 | Yes |
| int32 | Yes (not tested; in GetNCCLType switch) |
| int64 | Yes (not tested; in GetNCCLType switch) |
| int8 (signed) | **NO** — `GetNCCLType()` in `kvstore_nccl.h` has no case for `mshadow::kInt8`; raises `MXNetError: Unknown type passed to NCCL KVStore` asynchronously at `waitall()` |

### Key constraints discovered

1. **All keys must be init()'d before any push().** Once `push()` is called,
   NCCL communicators are locked and adding new keys causes `!is_none()` error.
   This is a `KVStoreNCCL` implementation constraint, not a NCCL library limit.

2. **Async error propagation.** An unsupported dtype error (e.g. int8) is raised
   at `waitall()` not at `push()`. After such an error the NCCL stream is
   corrupted for the remainder of the process — subsequent push/pull on any
   kvstore returns zeroed results without raising. No public API to reset.

---

## Per-process / 1-proc-per-GPU (NOT built-in)

### Architecture

`KVStoreNCCL` is a **single-process** abstraction. For cross-process NCCL
all-reduce, the available paths are:

| Approach | In this wheel | Notes |
|---|---|---|
| `kv.create('nccl')` | Yes | Single-process only. N workers each get an isolated size-1 communicator. |
| `dist_sync` / `dist_device_sync` / `dist_async` | Yes | PS-Lite parameter server over TCP/RDMA. Does NOT use NCCL. |
| Horovod | No | External library; wraps MXNet tensors into `ncclAllReduce`. |
| BytePS | No | External library. |

### Verification — 3/3 PASS

| Test | Result |
|---|---|
| Per-process workers are isolated (each gets own value back, not reduced sum) | PASS |
| Single-process multi-GPU all-reduce is the supported pattern | PASS |
| Documentation test (architecture constraint) | PASS |

Run: `pytest tests/python/gpu/test_nccl_multiproc.py -v --timeout=300`

### What happens when each process calls `kv.create('nccl')`

Each spawned worker creates an independent `KVStoreNCCL` with `ncclCommInitAll()`
over its own set of visible devices. With one GPU per worker the communicator
has size 1 — push/pull is a no-op local copy. Workers do not communicate.
Verified: worker 0 pushes 1.0 and pulls 1.0 back; worker 1 pushes 2.0 and
pulls 2.0 back. No cross-process reduction occurs.

---

## Issue #10 resolution

**Status: PARTIALLY RESOLVED**

- Single-process / multi-GPU `kv.create('nccl')` push/pull: **CONFIRMED WORKING** (10/10 tests).
- Per-process / 1-proc-per-GPU NCCL all-reduce: **NOT SUPPORTED** by MXNet's
  built-in kvstore layer. Requires Horovod or BytePS. Behaviour documented and
  tested: each worker is independently isolated. Issues.md updated accordingly.

---

## Test files

- `tests/python/gpu/test_nccl_singleproc.py` — 10 tests, single-process, 2 GPU
- `tests/python/gpu/test_nccl_multiproc.py` — 3 tests, spawned workers, isolation proof
