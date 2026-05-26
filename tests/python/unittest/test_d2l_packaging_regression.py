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

"""Regression tests for packaging issues found by the d2l-neu build.

Covers two distinct user-visible failures from the d2l-neu 2026-05-23
wheel-acceptance run:

- Issue 4: ``mxnet.__version__`` reported the stale literal
  ``2.0.0+cu13.bw.20260518.1`` for every wheel built since, even though
  the wheel METADATA was correct.  ``python/setup.py`` now writes the
  resolved ``MXNET_PACKAGE_VERSION`` into ``mxnet/_build_info.py`` at
  build time, and ``libinfo.py`` imports it.

- Issue 5: a ``Using Pooled (Naive) StorageManager for GPU/CPU`` banner
  appeared on stderr on the first allocation in every device context,
  polluting d2l notebook output cells.  It is now gated behind
  ``MXNET_LOG_STORAGE_INIT``.

These tests only meaningfully run against an installed wheel — they are
no-ops in editable/source-tree imports where neither the
``_build_info.py`` shim nor the bundled libmxnet apply.
"""

import importlib.metadata
import os
import subprocess
import sys

import pytest

import mxnet as mx


def _is_installed_wheel():
    """True when mxnet was imported from a site-packages install (not editable)."""
    pkg_dir = os.path.dirname(mx.__file__)
    return "site-packages" in pkg_dir


def test_version_matches_wheel_metadata():
    """`mxnet.__version__` must match the installed wheel's METADATA version."""
    if not _is_installed_wheel():
        pytest.skip(
            "source-tree/editable install: wheel METADATA version is unavailable")
    try:
        wheel_version = importlib.metadata.version("mxnet")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("mxnet wheel metadata unavailable in this environment")
    assert mx.__version__ == wheel_version, (
        "mxnet.__version__ ({}) disagrees with the installed wheel's METADATA "
        "Version ({}).  setup.py should write `_build_info.py` at build time so "
        "the in-Python value tracks MXNET_PACKAGE_VERSION."
        .format(mx.__version__, wheel_version)
    )


def test_first_allocation_does_not_print_storage_manager_banner():
    """No `Using ... StorageManager for ...` line on stderr at default verbosity.

    Forks a clean subprocess (so prior tests in the session don't pollute the
    static-init state) and watches stderr.  The banner was the single most
    visible artifact in d2l notebook outputs; if it returns, this test fires.
    """
    script = (
        "import os, sys\n"
        # Make sure the gate env var is not set to something truthy.
        "os.environ.pop('MXNET_LOG_STORAGE_INIT', None)\n"
        "import mxnet as mx\n"
        # Force a first allocation in the CPU context to trigger the storage
        # manager init path.
        "_ = mx.nd.ones((2, 3)).sum().asscalar()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        "subprocess crashed (rc={}):\nSTDOUT:\n{}\nSTDERR:\n{}"
        .format(proc.returncode, proc.stdout, proc.stderr))
    combined = proc.stdout + proc.stderr
    assert "StorageManager for" not in combined, (
        "Storage manager banner leaked to stderr/stdout at default verbosity. "
        "It is supposed to be gated behind MXNET_LOG_STORAGE_INIT=1.  Captured:\n"
        "{}".format(combined)
    )


def test_storage_manager_banner_opt_in_with_env_flag():
    """With MXNET_LOG_STORAGE_INIT=1, the banner *is* expected to appear."""
    script = (
        "import os\n"
        "os.environ['MXNET_LOG_STORAGE_INIT'] = '1'\n"
        "import mxnet as mx\n"
        "_ = mx.nd.ones((2, 3)).sum().asscalar()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    assert "StorageManager for" in combined, (
        "Expected the storage manager banner to appear with "
        "MXNET_LOG_STORAGE_INIT=1, but it was missing.  Captured:\n"
        "{}".format(combined)
    )


def test_import_does_not_trigger_numpy_subnormal_warning():
    """Importing MXNet must not leave NumPy's finfo probe in FTZ/DAZ mode."""
    script = (
        "import mxnet as mx\n"
        "import numpy as np\n"
        "_ = np.finfo(np.float32).smallest_subnormal\n"
        "_ = np.finfo(np.float64).smallest_subnormal\n"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "default", "-c", script],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        "subprocess crashed (rc={}):\nSTDOUT:\n{}\nSTDERR:\n{}"
        .format(proc.returncode, proc.stdout, proc.stderr))
    assert "smallest subnormal" not in proc.stderr, (
        "MXNet import changed the Python thread's floating-point mode before "
        "NumPy finfo was initialized.  Captured stderr:\n{}"
        .format(proc.stderr)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
