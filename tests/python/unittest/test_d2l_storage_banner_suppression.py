# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""d2l-mxnet-issues.md Issue 5 — pin the storage-manager banner suppression.

Before the fix, src/storage/storage.cc:208 emitted

  [HH:MM:SS] /home/.../src/storage/storage.cc:202: Using Pooled (Naive)
             StorageManager for cpu(0)
  [HH:MM:SS] /home/.../src/storage/storage.cc:202: Using Pooled (Naive)
             StorageManager for gpu(0)

unconditionally on first allocation per device.  This polluted every d2l
notebook's first output cell with timestamped internal-source paths and
broke output-deduplication tooling.

The fix gates the line behind `MXNET_LOG_STORAGE_INIT=1`.  This test pins
both behaviours: off by default, on when explicitly requested.
"""

import os
import subprocess
import sys
import textwrap

import pytest


def _spawn(snippet, env=None):
    env_ = dict(os.environ)
    # The banner line is opt-in; isolate from any caller env.
    env_.pop("MXNET_LOG_STORAGE_INIT", None)
    if env:
        env_.update(env)
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True, env=env_, timeout=180)


_TOUCH_BOTH = textwrap.dedent("""
    import mxnet as mx
    _ = mx.nd.zeros((1,), ctx=mx.cpu())
    try:
        if mx.context.num_gpus() > 0:
            _g = mx.nd.zeros((1,), ctx=mx.gpu(0))
    except Exception:
        pass
""")


def test_storage_banner_default_quiet():
    """At default verbosity, the storage banner must not appear on
    either stdout or stderr.  d2l notebooks rely on this to keep
    output cells clean."""
    r = _spawn(_TOUCH_BOTH)
    combined = r.stdout + "\n" + r.stderr
    assert "StorageManager for" not in combined, (
        "Storage banner regressed — appears at default verbosity. "
        f"stdout:\n{r.stdout}\n\nstderr:\n{r.stderr}")


def test_storage_banner_opt_in_visible():
    """With MXNET_LOG_STORAGE_INIT=1, the banner must still be available
    so storage-manager debugging is not lost."""
    r = _spawn(_TOUCH_BOTH, env={"MXNET_LOG_STORAGE_INIT": "1"})
    combined = r.stdout + "\n" + r.stderr
    assert "StorageManager for" in combined, (
        "MXNET_LOG_STORAGE_INIT=1 must surface the banner. "
        f"stdout:\n{r.stdout}\n\nstderr:\n{r.stderr}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
