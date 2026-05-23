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

"""GH1 tail: source-grep regression for Dockerfile + workflow hardening.

A prior GH1 sweep identified URL fetchers without integrity checks, stale
GitHub Action versions, and a DNS-resolve path with unbounded timeout.
This file pins each of those fixes so a future maintainer who refactors
the surrounding code doesn't silently regress the security posture.

It does not run docker / network — only checks file contents.
"""

import os
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]


def _read(rel):
    p = REPO / rel
    assert p.exists(), f"expected file missing: {p}"
    return p.read_text()


def test_android_ndk_dockerfile_pins_sha256():
    contents = _read("ci/docker/Dockerfile.build.android")
    # The Google-published NDK r19c hash must remain pinned; sha256sum -c
    # is what gates the toolchain installation.
    assert "4f61cbe4bbf6406aa5ef2ae871def78010eed6271af72de83f8bd0b07a9fd3fd" in contents, \
        "GH1: NDK r19c SHA256 pin removed from Dockerfile.build.android"
    assert "sha256sum -c" in contents, \
        "GH1: NDK Dockerfile no longer verifies checksum"
    assert "--fail" in contents, \
        "GH1: NDK curl must use --fail to abort on HTTP error"


def test_arm_toolchain_dockerfile_uses_defensive_flags():
    contents = _read("ci/docker/Dockerfile.build.arm")
    # The bootlin tarball has no upstream-supplied SHA on a stable mirror,
    # so a BUILD ARG is exposed for site-local pinning. The defensive curl
    # flags must remain.
    assert "ARG ARMV6_TOOLCHAIN_SHA256" in contents, \
        "GH1: ARMv6 toolchain SHA256 build arg removed"
    assert "--fail" in contents and "--proto '=https'" in contents and "--tlsv1.2" in contents, \
        "GH1: ARMv6 toolchain curl missing defensive flags"


def test_get_pip_download_is_strict():
    contents = _read("cd/python/docker/Dockerfile")
    # get-pip.py is a moving target on bootstrap.pypa.io; we cannot pin
    # SHA256, but we must use TLSv1.2+, retries, timeout, and check that
    # the download produced a non-empty file before exec'ing it.
    assert "--secure-protocol=TLSv1_2" in contents and "--timeout=" in contents and "--tries=" in contents, \
        "GH1: get-pip.py wget missing TLS / timeout / retry flags"
    assert "test -s get-pip.py" in contents, \
        "GH1: get-pip.py exec is no longer gated by non-empty download check"


def test_deploy_sh_uses_defensive_wget():
    contents = _read("ci/publish/website/deploy.sh")
    assert "--secure-protocol=TLSv1_2" in contents and "--timeout=" in contents and "--tries=" in contents, \
        "GH1: deploy.sh wget missing TLS / timeout / retry flags"
    assert 'test -s "$api-artifacts.tgz"' in contents, \
        "GH1: deploy.sh no longer rejects empty artifact downloads"


def test_workflows_use_modern_action_versions():
    """The three previously-stale workflows (link_check, os_x_mklbuild,
    os_x_staticbuild) must use checkout@v4 + setup-python@v5 — matching
    the rest of the workflow set."""
    for wf in [
        ".github/workflows/link_check.yml",
        ".github/workflows/os_x_mklbuild.yml",
        ".github/workflows/os_x_staticbuild.yml",
    ]:
        contents = _read(wf)
        # We don't pin to exactly v4/v5 (a future bump should be allowed),
        # but @v2/@v3 must not return.
        for stale in ["actions/checkout@v2", "actions/checkout@v3",
                      "actions/setup-python@v2", "actions/setup-python@v3"]:
            assert stale not in contents, \
                f"GH1: {wf} regressed to {stale}"


def test_diagnose_py_bounds_dns_timeout():
    contents = _read("tools/diagnose.py")
    # `gethostbyname` has no per-call timeout; the fix sets+restores
    # the process-wide default so a hung DNS resolver doesn't block
    # `mxnet.tools.diagnose` forever.
    assert "socket.setdefaulttimeout(timeout)" in contents, \
        "GH1: diagnose.py DNS resolve is no longer bounded"
    assert "socket.setdefaulttimeout(prev_default)" in contents, \
        "GH1: diagnose.py DNS resolve does not restore previous default"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
