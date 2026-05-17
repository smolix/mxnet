# Bug report: gluon-data / image test files

Investigation date: 2026-05-17 (HEAD `8a47e5a9a` -> fix commit `7934d40d7`)

## TL;DR
None of the three target test files crashes with SIGSEGV / exit 134 any more.
Each file passes when run in isolation. The earlier "exit 134" reports were
likely older crashes that have been resolved by the intervening port commits
(CUDA 13, cuDNN 9, oneDNN v3, batchnorm port). What remains are two pure-Python
bugs:

1. **FIXED (committed as `7934d40d7`):**
   `batchify.Stack` blew up with
   `TypeError: dtype must be one of ...` on every multi-worker DataLoader that
   yielded legacy `mx.nd.NDArray` samples while `is_np_array()` was True.
2. **NOT FIXED (pre-existing, separate from the target):**
   `tests/python/unittest/test_image.py` calls `mx.npx.reset_np()` at module
   scope (line 29). This is a *global, process-wide* side effect that flips the
   gluon stack out of numpy semantics for every test file that loads after it
   in the same pytest invocation, which then makes other tests fail with
   `'NDArray' object has no attribute 'item'` etc.

## Per-file status

| file                                       | isolated run | combined run | first failing test                         |
|--------------------------------------------|--------------|--------------|--------------------------------------------|
| `tests/python/unittest/test_image.py`      | 14p / 4s     | 14p / 4s     | -                                           |
| `tests/python/unittest/test_contrib_gluon_data_vision.py` | 3p | 3p | - |
| `tests/python/unittest/test_gluon_data.py` | 30p          | 28p / 2 fail | `test_recordimage_dataset` (only when run AFTER `test_image.py`) |

`p` = passed, `s` = skipped. All runs used `--timeout=300`.

No SIGSEGV / SIGABRT / "exit 134" observed in any configuration. faulthandler
was always enabled (`python -X faulthandler`); no native backtraces were
emitted.

## Fix details (commit `7934d40d7`)

`python/mxnet/gluon/data/batchify.py::Stack.__call__` ignored
`mx.nd.NDArray` samples when `is_np_array()` was True, because the
`isinstance(data[0], _arr_cls)` check used `_arr_cls = mx.np.ndarray`. The
fallback branch then called `np.asarray(list_of_NDArrays)`, which numpy 2.x
turns into a `dtype=object` array. That object dtype was passed back into
`_new_alloc_handle -> dtype_np_to_mx`, which raises `TypeError`.

Repro:
```python
import mxnet as mx
from mxnet import gluon

class D(gluon.data.Dataset):
    def __len__(self): return 100
    def __getitem__(self, i): return mx.nd.full((10,), i)

for batch in gluon.data.DataLoader(D(), batch_size=1, num_workers=5):
    pass
# -> TypeError: dtype must be one of: {None: -1, np.float32: 0, ...}
```

Fix: convert legacy `nd.NDArray` samples to the np-mode array via the
zero-copy `as_np_ndarray()` before the stacking branch decides what to do.
This routes through `mx.np.stack` instead of the `np.asarray` fallback, which
is what `default_batchify_fn` / `default_mp_batchify_fn` (in `dataloader.py`)
already do.

## Pre-existing pollution bug (not fixed here)

```
tests/python/unittest/test_image.py:29:
    mx.npx.reset_np()
```

Running this at module top-level means *importing* `test_image.py` (which
pytest does during collection if it appears in the argv) is enough to
permanently disable np semantics for the rest of the process. Symptoms in
combined runs:

* `test_gluon_data.py::test_recordimage_dataset` fails with
  `AttributeError: 'NDArray' object has no attribute 'item'` because
  `mx.nd.full(...)` no longer yields an `np.ndarray`.
* Same symptom in `test_recordimage_dataset_with_data_loader_multiworker`.

### Recommended fix
Move the `reset_np()` call into a pytest fixture with `autouse=True` and a
matching `mx.npx.set_np()` in the teardown, e.g.:

```python
@pytest.fixture(autouse=True)
def _classic_mxnet_scope():
    mx.npx.reset_np()
    try:
        yield
    finally:
        mx.npx.set_np()
```

Or, less invasively, mark the file so it must run last (`-p no:randomly`,
ordering, or `pytest.mark.order`).

## What to do next
* The committed fix (`7934d40d7`) is independently correct and should remain.
* When you get back to data-loader hygiene, also clean up the `reset_np()`
  global side effect in `test_image.py`. After that, the three files can run
  in the same pytest invocation without test ordering surprises.
* No native debugger session was needed; `python -X faulthandler` was always
  silent. If a real SIGSEGV reappears in CI, the cleanest next step is
  `gdb --args python -m pytest -x <file>` plus `set follow-fork-mode child`
  to catch crashes in DataLoader worker processes (multiprocessing fork).
