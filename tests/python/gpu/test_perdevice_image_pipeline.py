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

"""ThreadedEnginePerDevice image-pipeline regression tests.

The D2L hotdog pipeline exposed a host-side out-of-bounds read in
RandomResizedCrop's fallback path while PerDevice workers were feeding GPU
training.  Run each GPU in its own subprocess so each worker gets a fresh MXNet
engine/context and the test can scale without one long serial process.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import mxnet as mx


pytestmark = pytest.mark.skipif(mx.context.num_gpus() == 0, reason="requires GPU")


_WORKER = r"""
import gc
import sys

import mxnet as mx
from mxnet import np, npx

npx.set_np()

if not mx.runtime.Features().is_enabled("OPENCV"):
    print("SKIP_OPENCV")
    sys.exit(0)

dev_id = int(sys.argv[1])
ctx = mx.gpu(dev_id)
x = np.ones((4, 50, 100, 3), dtype="uint8") * 11

def used_bytes():
    free, total = mx.context.gpu_memory_info(dev_id)
    return total - free

def drain():
    mx.npx.waitall()
    gc.collect()
    ctx.empty_cache()
    mx.npx.waitall()

def step():
    y = npx.image.random_resized_crop(x, width=224, height=224, max_trial=0)
    y = npx.image.random_flip_left_right(y, p=1.0)
    y = npx.image.to_tensor(y)
    yg = y.to_device(ctx)
    loss = (yg * 1.1).sum()
    loss.wait_to_read()
    del y, yg, loss

for _ in range(4):
    step()
drain()
before = used_bytes()

for _ in range(24):
    step()
    drain()

after = used_bytes()
growth_mib = (after - before) / (1 << 20)
print(f"device={dev_id} before={before} after={after} growth_mib={growth_mib:.1f}")
assert after - before <= 256 * (1 << 20), (
    f"GPU memory grew by {growth_mib:.1f} MiB in PerDevice image pipeline"
)
"""


@pytest.mark.timeout(240)
def test_perdevice_image_pipeline_gpu_workers_do_not_crash_or_leak():
    if not mx.runtime.Features().is_enabled("OPENCV"):
        pytest.skip("requires OpenCV image operators")

    worker_count = min(mx.context.num_gpus(), 4)
    env = os.environ.copy()
    env.update({
        "MXNET_ENGINE_TYPE": "ThreadedEnginePerDevice",
        "MXNET_CPU_WORKER_NTHREADS": "2",
        "MXNET_GPU_WORKER_NTHREADS": "1",
        "MXNET_GPU_COPY_NTHREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(_WORKER), str(dev_id)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for dev_id in range(worker_count)
    ]
    failures = []
    for dev_id, proc in enumerate(procs):
        out, err = proc.communicate(timeout=240)
        if proc.returncode != 0:
            failures.append(
                f"device {dev_id} rc={proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
            )

    assert not failures, "\n\n".join(failures)
