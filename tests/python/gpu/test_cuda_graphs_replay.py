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

"""CUDA Graphs differential-replay correctness net (CUDA_GRAPHS_PLAN.md Phase 1).

MXNET_CUDA_GRAPHS_VERIFY=1 makes the graph executor, for every captured subseg,
run the ops both as a captured graph and conventionally (from identical
pre-segment state) and assert the outputs match — LOG(FATAL) on divergence. A
divergence aborts the process, so these tests run the workload in a subprocess
and assert a clean exit plus the expected log markers.
"""
import os
import subprocess
import sys

import pytest

# A deterministic, hybridized static block whose ops are CUDA-graph-capturable
# (conv + relu). Run under graphs + verify; the net must confirm graph ==
# conventional for every captured subseg.
_WORKLOAD = r"""
import mxnet as mx
from mxnet import np, npx, gluon
from mxnet.gluon import nn
npx.set_np()
dev = mx.gpu(0)
net = nn.HybridSequential()
for _ in range(6):
    net.add(nn.Conv2D(channels=16, kernel_size=3, padding=1, activation="relu"))
net.initialize(device=dev)
net.hybridize(static_alloc=True, static_shape=True)
x = np.ones((8, 16, 16, 16), device=dev)
for _ in range(20):
    y = net(x)
y.wait_to_read()
npx.waitall()
print("WORKLOAD_OK")
"""

# FullyConnected (cuBLAS gemm) capture via cuBLASLt — parametrized by dtype so
# both fp32 and fp16 FC capture are regression-tested (CUDA_GRAPHS_PLAN.md
# Phase 2). dtype is substituted in.
_FC_WORKLOAD = r"""
import mxnet as mx
from mxnet import np, npx, gluon
from mxnet.gluon import nn
npx.set_np()
dev = mx.gpu(0)
net = nn.HybridSequential()
for _ in range(6):
    net.add(nn.Dense(256, activation="relu"))
net.initialize(device=dev)
net.cast("{dtype}")
net.hybridize(static_alloc=True, static_shape=True)
x = np.ones((32, 256), device=dev, dtype="{dtype}")
for _ in range(20):
    y = net(x)
y.wait_to_read()
npx.waitall()
print("WORKLOAD_OK")
"""


# batch_dot / matmul (cuBLASLt batched gemm) capture — Phase 2b. Both route their
# GPU gemm through linalg_batch_gemm and default to full fp32 (PyTorch parity).
_BATCHDOT_WORKLOAD = r"""
import mxnet as mx
from mxnet import np, npx, gluon
npx.set_np()
dev = mx.gpu(0)
class Net(gluon.HybridBlock):
    def __init__(s, n, **k):
        super().__init__(**k); s.w = gluon.Parameter('w', shape=(4, n, n))
    def forward(s, x):
        w = s.w.data()
        for _ in range(4):
            x = npx.batch_dot(x, w); x = npx.relu(x)
        return x
net = Net(64); net.initialize(device=dev)
net.hybridize(static_alloc=True, static_shape=True)
x = np.ones((4, 64, 64), device=dev) * 0.01
for _ in range(20):
    y = net(x)
y.wait_to_read(); npx.waitall(); print("WORKLOAD_OK")
"""

_MATMUL_WORKLOAD = r"""
import mxnet as mx
from mxnet import np, npx, gluon
npx.set_np()
dev = mx.gpu(0)
class Net(gluon.HybridBlock):
    def __init__(s, n, **k):
        super().__init__(**k); s.w = gluon.Parameter('w', shape=(4, n, n))
    def forward(s, x):
        w = s.w.data()
        for _ in range(4):
            x = np.matmul(x, w); x = npx.relu(x)
        return x
net = Net(64); net.initialize(device=dev)
net.hybridize(static_alloc=True, static_shape=True)
x = np.ones((4, 64, 64), device=dev) * 0.01
for _ in range(20):
    y = net(x)
y.wait_to_read(); npx.waitall(); print("WORKLOAD_OK")
"""


# End-to-end training: a hybridized static MLP + SGD trained a few steps. Prints
# the per-step loss trajectory as JSON so the test can assert graphs-on ==
# graphs-off (captures FC forward+backward + the optimizer step).
_TRAIN_WORKLOAD = r"""
import json
import mxnet as mx
from mxnet import np, npx, gluon, autograd
from mxnet.gluon import nn
npx.set_np()
dev = mx.gpu(0)
mx.np.random.seed(7)
net = nn.HybridSequential()
for _ in range(4):
    net.add(nn.Dense(128, activation='relu'))
net.add(nn.Dense(10))
net.initialize(mx.init.Xavier(), device=dev)
net.hybridize(static_alloc=True, static_shape=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
mx.np.random.seed(123)
X = np.random.uniform(size=(64, 256), device=dev)
Y = np.random.randint(0, 10, size=(64,), device=dev).astype('float32')
losses = []
for _ in range(30):
    with autograd.record():
        l = loss_fn(net(X), Y).mean()
    l.backward(); trainer.step(1)
    losses.append(round(float(l.item()), 5))
print("LOSSES", json.dumps(losses))
"""


def _run(env_extra, code=_WORKLOAD):
    env = os.environ.copy()
    env["MXNET_ENABLE_CUDA_GRAPHS"] = "1"
    env["MXNET_USE_FUSION"] = "0"
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code],
                          env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.serial
def test_cuda_graphs_differential_replay_matches():
    """Deterministic captured graphs must match conventional execution exactly."""
    r = _run({"MXNET_CUDA_GRAPHS_VERIFY": "1", "MXNET_CUDA_GRAPHS_VERBOSE": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"verify aborted (divergence?):\n{out[-3000:]}"
    assert "WORKLOAD_OK" in r.stdout, f"workload did not finish:\n{out[-3000:]}"
    assert "differential-replay MISMATCH" not in out, f"unexpected mismatch:\n{out[-3000:]}"
    # Proof the net actually engaged on captured subsegs.
    assert "replay OK" in out, f"verify net did not run; no 'replay OK':\n{out[-3000:]}"


@pytest.mark.serial
@pytest.mark.parametrize("dtype,rtol,atol", [("float32", "1e-3", "1e-4"),
                                             ("float16", "1e-2", "1e-2")])
def test_cuda_graphs_fc_cublaslt_capture(dtype, rtol, atol):
    """FullyConnected (cuBLAS gemm) captures via cuBLASLt and matches conventional.

    Exercises the Phase-2 capture-safe cuBLASLt path for both fp32 and fp16.
    """
    r = _run({"MXNET_CUDA_GRAPHS_ALLOW_CUBLAS": "1",
              "MXNET_CUDA_GRAPHS_VERIFY": "1",
              "MXNET_CUDA_GRAPHS_VERBOSE": "1",
              "MXNET_CUDA_GRAPHS_VERIFY_RTOL": rtol,
              "MXNET_CUDA_GRAPHS_VERIFY_ATOL": atol},
             code=_FC_WORKLOAD.format(dtype=dtype))
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"FC capture aborted ({dtype}):\n{out[-3000:]}"
    assert "WORKLOAD_OK" in r.stdout, f"workload did not finish ({dtype}):\n{out[-3000:]}"
    assert "differential-replay MISMATCH" not in out, f"mismatch ({dtype}):\n{out[-3000:]}"
    # The capture-unsafe legacy path must never be reached during capture.
    assert "capture-unsafe legacy cuBLAS" not in out, f"hit legacy fallback ({dtype}):\n{out[-3000:]}"
    # FC must actually be inside a captured graph (not bypassed).
    assert "FullyConnected" in out and "replay OK" in out, \
        f"FC not captured/verified ({dtype}):\n{out[-3000:]}"


@pytest.mark.serial
@pytest.mark.parametrize("name,code", [("batch_dot", _BATCHDOT_WORKLOAD),
                                       ("matmul", _MATMUL_WORKLOAD)],
                         ids=["batch_dot", "matmul"])
def test_cuda_graphs_batched_gemm_capture(name, code):
    """batch_dot / matmul capture via linalg_batch_gemm (cuBLASLt) and match conventional."""
    r = _run({"MXNET_CUDA_GRAPHS_ALLOW_CUBLAS": "1",
              "MXNET_CUDA_GRAPHS_VERIFY": "1",
              "MXNET_CUDA_GRAPHS_VERBOSE": "1"},
             code=code)
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"{name} capture aborted:\n{out[-3000:]}"
    assert "WORKLOAD_OK" in r.stdout, f"{name} did not finish:\n{out[-3000:]}"
    assert "differential-replay MISMATCH" not in out, f"{name} mismatch:\n{out[-3000:]}"
    assert "capture-unsafe legacy cuBLAS" not in out, f"{name} hit legacy fallback:\n{out[-3000:]}"
    assert "replay OK" in out, f"{name} not captured/verified:\n{out[-3000:]}"


@pytest.mark.serial
def test_cuda_graphs_training_matches_eager():
    """End-to-end training (FC fwd+bwd captured + SGD) must match graphs-off exactly."""
    import json

    def losses(env_extra):
        r = _run(env_extra, code=_TRAIN_WORKLOAD)
        out = r.stdout + r.stderr
        assert r.returncode == 0, f"train run failed:\n{out[-3000:]}"
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("LOSSES")][0]
        return json.loads(line[len("LOSSES"):])

    off = losses({"MXNET_ENABLE_CUDA_GRAPHS": "0"})
    on = losses({"MXNET_CUDA_GRAPHS_ALLOW_CUBLAS": "1"})  # graphs on via _run default
    assert len(off) == len(on) == 30
    for i, (a, b) in enumerate(zip(off, on)):
        assert abs(a - b) <= 1e-4, f"step {i}: graphs-off {a} vs graphs-on {b}\noff={off}\non={on}"


@pytest.mark.serial
def test_cuda_graphs_capture_summary_emitted():
    """Verbose mode emits the per-segment capture summary (Phase 1 observability)."""
    r = _run({"MXNET_CUDA_GRAPHS_VERBOSE": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "CUDA graph segment summary" in out, f"no capture summary emitted:\n{out[-3000:]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
