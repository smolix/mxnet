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
    """No workflow may regress to a legacy action major (checkout@v2/v3,
    setup-python@v2/v3).

    Globs the live ``.github/workflows`` set instead of a hardcoded file
    list: an earlier hardcoded list (link_check, os_x_mklbuild,
    os_x_staticbuild) silently rotted into a missing-file failure when a CI
    cleanup deleted those workflows. Globbing tracks add/remove automatically.
    """
    wf_dir = REPO / ".github" / "workflows"
    workflows = sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml"))
    assert workflows, f"no workflow files found under {wf_dir}"
    # We don't pin to exactly v4/v5 (a future bump should be allowed),
    # but @v2/@v3 must not return.
    for wf in workflows:
        contents = wf.read_text()
        for stale in ["actions/checkout@v2", "actions/checkout@v3",
                      "actions/setup-python@v2", "actions/setup-python@v3"]:
            assert stale not in contents, \
                f"GH1: {wf.relative_to(REPO)} regressed to {stale}"


def test_diagnose_py_bounds_dns_timeout():
    contents = _read("tools/diagnose.py")
    # `gethostbyname` has no per-call timeout; the fix sets+restores
    # the process-wide default so a hung DNS resolver doesn't block
    # `mxnet.tools.diagnose` forever.
    assert "socket.setdefaulttimeout(timeout)" in contents, \
        "GH1: diagnose.py DNS resolve is no longer bounded"
    assert "socket.setdefaulttimeout(prev_default)" in contents, \
        "GH1: diagnose.py DNS resolve does not restore previous default"


def test_diagnose_py_bounds_sysctl_timeout():
    contents = _read("tools/diagnose.py")
    assert "communicate(timeout=10)" in contents, \
        "diagnose.py sysctl call is no longer bounded"
    assert "pipe.kill()" in contents, \
        "diagnose.py sysctl timeout no longer kills the child process"


def test_example_gluon_data_uses_context_managed_tarfile():
    contents = _read("example/gluon/data.py")
    assert 'with tarfile.open(tar_path, "r:gz") as tar:' in contents, \
        "example/gluon/data.py no longer closes tarfile on extraction failure"
    assert "_safe_extract_tar(tar, data_folder)" in contents, \
        "example/gluon/data.py no longer validates tar members before extraction"


def test_example_zip_extractors_validate_members():
    horovod = _read("example/distributed_training-horovod/gluon_mnist.py")
    super_resolution = _read("example/gluon/super_resolution/super_resolution.py")
    assert "https://data.mxnet.io/mxnet/data/mnist.zip" in horovod, \
        "Horovod MNIST example reintroduced a plaintext dataset download"
    assert "_safe_extract_zip(zf, data_dir)" in horovod and "commonpath" in horovod, \
        "Horovod MNIST example no longer validates zip members before extraction"
    assert "_safe_extract_zip(archive, tmp_dir)" in super_resolution and "commonpath" in super_resolution, \
        "Super-resolution example no longer validates zip members before extraction"


def test_gluon_super_resolution_closes_pil_images():
    data_py = _read("example/gluon/data.py")
    super_resolution = _read("example/gluon/super_resolution/super_resolution.py")

    assert "with Image.open(fn) as img:" in data_py
    assert "image = img.convert('YCbCr').split()[0]" in data_py
    assert "with Image.open(opt.resolve_img) as img:" in super_resolution
    assert "y, cb, cr = img.convert('YCbCr').split()" in super_resolution


def test_gluon_super_resolution_rejects_partial_bsds500_cache():
    super_resolution = _read("example/gluon/super_resolution/super_resolution.py")

    assert "def _dataset_ready():" in super_resolution
    assert 'path.join(data_dir, "images", "train")' in super_resolution
    assert 'path.join(data_dir, "groundTruth", "test")' in super_resolution
    assert "if _dataset_ready():" in super_resolution
    assert "if path.exists(data_dir):\n            shutil.rmtree(data_dir)" in super_resolution
    assert "except Exception:" in super_resolution and "raise" in super_resolution
    assert "finally:" in super_resolution and "shutil.rmtree(datasets_tmpdir)" in super_resolution


def test_download_and_sparse_benchmark_avoid_assert_and_shell():
    test_utils = _read("python/mxnet/test_utils.py")
    sparse_dot = _read("benchmark/python/sparse/dot.py")
    sparse_op = _read("benchmark/python/sparse/sparse_op.py")
    assert "assert r.status_code == 200" not in test_utils, \
        "download helper must not rely on assert for HTTP status validation"
    assert "raise RuntimeError(f\"failed to open {url}: HTTP {r.status_code}\")" in test_utils, \
        "download helper no longer raises an explicit HTTP status error"
    assert "shell=True" not in sparse_dot and "os.system" not in sparse_dot, \
        "sparse dot benchmark reintroduced shell command execution"
    assert "os.system" not in sparse_op, \
        "sparse op benchmark reintroduced shell command execution"


def test_quantization_examples_use_https_datasets():
    inference = _read("example/quantization/imagenet_inference.py")
    qsym = _read("example/quantization/imagenet_gen_qsym_onednn.py")
    assert "https://data.mxnet.io/data/val_256_q90.rec" in inference
    assert "http://data.mxnet.io/data/val_256_q90.rec" not in inference
    assert "https://data.mxnet.io/data/val_256_q90.rec" in qsym
    assert "http://data.mxnet.io/data/val_256_q90.rec" not in qsym


def test_builtin_dataset_extractors_validate_archive_members():
    datasets = _read("python/mxnet/gluon/data/vision/datasets.py")
    test_utils = _read("python/mxnet/test_utils.py")
    assert "_safe_extract_tar(tar, self._root)" in datasets and "commonpath" in datasets, \
        "Gluon vision datasets no longer validate tar members before extraction"
    assert "_safe_extract_zip(zf, path)" in test_utils and "commonpath" in test_utils, \
        "test_utils CIFAR downloader no longer validates zip members before extraction"


def test_movielens_data_avoids_shell_download_and_validates_zip():
    contents = _read("example/recommenders/movielens_data.py")
    assert "os.system" not in contents, \
        "MovieLens downloader reintroduced shell command execution"
    assert "_safe_extract_zip" in contents and "startswith(os.pardir + os.sep)" in contents, \
        "MovieLens downloader no longer validates zip members before extraction"


def test_movielens_data_rejects_partial_cache():
    contents = _read("example/recommenders/movielens_data.py")
    assert "def _dataset_ready(prefix):" in contents
    assert '"u1.base", "u1.test"' in contents
    assert "if not _dataset_ready(prefix):" in contents
    assert "shutil.rmtree(prefix)" in contents
    assert "MovieLens dataset extraction incomplete" in contents


def test_im2rec_cleans_up_worker_processes_on_failure():
    contents = _read("tools/im2rec.py")
    assert "finally:" in contents and "p.terminate()" in contents and "p.kill()" in contents, \
        "im2rec multiprocessing path no longer cleans up live children on failure"
    assert "q.close()" in contents and "q_out.close()" in contents, \
        "im2rec multiprocessing queues are no longer closed on failure"


def test_cpp_imagenet_inference_downloads_are_bounded_and_tar_checked():
    contents = _read("cpp-package/example/inference/unit_test_imagenet_inference.sh")
    assert "--timeout=30" in contents and "--tries=3" in contents and "--secure-protocol=TLSv1_2" in contents, \
        "C++ imagenet inference test wget calls no longer use bounded TLS download flags"
    assert "https://data.mxnet.io" in contents and "http://data.mxnet.io" not in contents, \
        "C++ imagenet inference test reintroduced plaintext data.mxnet.io downloads"
    assert "tar -tzf inception-bn.tar.gz" in contents and "Unsafe path" in contents, \
        "C++ imagenet inference test no longer validates tar members before extraction"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
