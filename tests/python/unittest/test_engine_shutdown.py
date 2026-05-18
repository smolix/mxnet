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

"""
Engine shutdown / lifecycle tests.

Addresses:
  A13 (apache/mxnet#11163) - DLL unload / static-dtor deadlock
  A7  (apache/mxnet#18090) - CI hang after last test completes
  A6  (apache/mxnet#19994) - WaitForVar hangs in long-running inference
"""

import subprocess
import sys
import os
import pytest


_VENV_PYTHON = sys.executable


def _run(cmd, env=None, timeout=60):
    """Run a subprocess and return (returncode, stdout+stderr combined)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged_env,
    )
    return result.returncode, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# A13: clean exit after basic ndarray operations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trial", range(5))
def test_clean_exit_basic(trial):
    """
    A13: import mxnet, do an ndarray multiply, print result, exit cleanly.
    No segfault, no hang (enforced by subprocess timeout).
    Run 5 times to flush transient flakes.
    """
    code = (
        "import mxnet as mx; "
        "x = mx.nd.array([1., 2.]); "
        "y = x * 2; "
        "print(y.asnumpy())"
    )
    rc, out = _run([_VENV_PYTHON, "-c", code], timeout=60)
    assert rc == 0, f"trial={trial} exited with code {rc}.\nOutput:\n{out}"


@pytest.mark.parametrize("trial", range(3))
def test_clean_exit_gpu(trial):
    """
    A13: same test but on GPU context if available.
    """
    code = "\n".join([
        "import mxnet as mx",
        "try:",
        "    x = mx.nd.array([1., 2.], ctx=mx.gpu(0))",
        "    y = (x * 3).asnumpy()",
        "    print(y)",
        "except mx.MXNetError:",
        "    print('no gpu, skip')",
    ])
    rc, out = _run([_VENV_PYTHON, "-c", code], timeout=60)
    assert rc == 0, f"trial={trial} exited with code {rc}.\nOutput:\n{out}"


# ---------------------------------------------------------------------------
# A7: CI-style hang — many ops then exit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trial", range(3))
def test_clean_exit_after_many_ops(trial):
    """
    A7: simulate a CI job that pushes many ops then exits.
    Previously engines could hang for hours after the last test.
    """
    code = (
        "import mxnet as mx\n"
        "import numpy as np\n"
        "ctx = mx.cpu()\n"
        "x = mx.nd.zeros((128, 128), ctx=ctx)\n"
        "for _ in range(200):\n"
        "    x = x + 1\n"
        "mx.nd.waitall()\n"
        "print('done', x.asnumpy()[0, 0])\n"
    )
    rc, out = _run([_VENV_PYTHON, "-c", code], timeout=120)
    assert rc == 0, f"trial={trial} exited with code {rc}.\nOutput:\n{out}"
    assert "done" in out, f"Expected 'done' in output.\nOutput:\n{out}"


# ---------------------------------------------------------------------------
# A6/A7 diagnostic mode: MXNET_ENGINE_DIAG=1 must not change behaviour
# ---------------------------------------------------------------------------

def test_diag_mode_smoke():
    """
    A6/A7: MXNET_ENGINE_DIAG=1 should not break normal execution.
    With MXNET_ENGINE_DIAG_TIMEOUT_S=5 (short for test speed) the process
    must still complete successfully without hitting the watchdog.
    """
    code = (
        "import mxnet as mx; "
        "x = mx.nd.array([1., 2.]) * 2; "
        "print(x.asnumpy())"
    )
    rc, out = _run(
        [_VENV_PYTHON, "-c", code],
        env={"MXNET_ENGINE_DIAG": "1", "MXNET_ENGINE_DIAG_TIMEOUT_S": "5"},
        timeout=60,
    )
    assert rc == 0, f"diag mode exited with code {rc}.\nOutput:\n{out}"
    # Must NOT have fired the watchdog warning
    assert "[MXNET_ENGINE_DIAG]" not in out, (
        "Watchdog fired during normal execution — engine is slower than expected.\n"
        f"Output:\n{out}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
