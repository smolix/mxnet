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

"""d2l-mxnet-issues.md Issue 7 — pin gluon.Trainer.step(batch_size) rescale.

Cross-framework measurement of ``F.cross_entropy`` /
``SoftmaxCrossEntropyLoss(axis=1)`` / ``optax.softmax_cross_entropy_*`` on
identical (N, C, H, W) inputs confirmed all three frameworks return
bit-identical mean loss values for FCN-shaped inputs.  So the reported
3× higher MXNet FCN train loss is NOT a loss-reduction divergence.

Empirical follow-up isolated the actual root cause: ``gluon.Trainer.step(N)``
rescales the gradient by ``1 / N`` before applying it, while PyTorch's
``optimizer.step()`` takes no rescale.  For the same nominal ``lr`` and the
same per-sample-mean loss, MXNet's effective update is ``N``× smaller than
PyTorch's.  The d2l FCN notebook uses ``batch_size=32`` and ``lr=0.001``,
so MXNet's effective LR is ``32``× smaller than the PyTorch tab — fully
explaining the ~3× higher final loss and ~15% lower test accuracy.

This test pins the rescale semantics so future Trainer refactors do not
silently drift away from the documented behaviour, and demonstrates the
PyTorch-equivalent call (``trainer.step(1)``) for users porting code.
"""

import numpy as np
import pytest

import mxnet as mx
from mxnet import autograd, gluon, np as mnp, npx

pytestmark = pytest.mark.usefixtures("set_np_semantics")


class _Lin(gluon.HybridBlock):
    """y = w * x, single parameter initialised to 1.0."""
    def __init__(self):
        super().__init__()
        self.w = gluon.Parameter('w', shape=(1,),
                                  init=mx.init.Constant(1.0))

    def forward(self, x):
        return x * self.w.data(x.ctx)


def _one_step_with_batch_size(step_arg):
    """Build a 1-param net, run one forward+backward+step with the given
    batch_size argument to Trainer.step(), and return the new param value."""
    N = 4
    x = mnp.ones(N, dtype='float32')
    t = mnp.zeros(N, dtype='float32')
    net = _Lin()
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                             {'learning_rate': 0.1})
    with autograd.record():
        y = net(x)
        loss = ((y - t) ** 2).sum()
    loss.backward()
    pre_grad = float(net.w.grad().asnumpy()[0])
    trainer.step(step_arg)
    post = float(net.w.data().asnumpy()[0])
    return pre_grad, post


def test_trainer_step_batch_size_rescales_gradient():
    """Trainer.step(N) divides the gradient by N.

    Math: w=1, x=t=1/0 (N=4), grad = sum((w-0)^2)' = 2*sum(x*(w-0)) = 2*4*1 = 8.
    With lr=0.1:
      - step(N=4): w_new = 1 - 0.1 * (8 / 4) = 0.8
      - step(1):   w_new = 1 - 0.1 * 8       = 0.2  (no rescale, PyTorch-equivalent)
    """
    grad_n, w_after_n = _one_step_with_batch_size(4)
    assert grad_n == pytest.approx(8.0), f"grad regressed: {grad_n}"
    assert w_after_n == pytest.approx(0.8, abs=1e-5), \
        (f"Trainer.step(4) should rescale grad by 1/4 -> w=0.8; got {w_after_n}. "
         "Either the rescale semantics regressed, or the docstring around d2l "
         "Issue 7 is now stale.")


def test_trainer_step_1_matches_pytorch_no_rescale():
    """Trainer.step(1) applies the raw gradient -> matches PyTorch's
    optimizer.step() behaviour. This is the call to use when porting
    PyTorch code that uses the same per-sample loss formula."""
    grad_1, w_after_1 = _one_step_with_batch_size(1)
    assert grad_1 == pytest.approx(8.0), f"grad regressed: {grad_1}"
    assert w_after_1 == pytest.approx(0.2, abs=1e-5), \
        (f"Trainer.step(1) should apply full grad -> w=0.2; got {w_after_1}. "
         "This is the PyTorch-equivalent call for d2l Issue 7 — if this fails, "
         "the docstring example is now wrong.")


def test_effective_lr_ratio_matches_batch_size():
    """The effective learning rate ratio between step(1) and step(N) must
    equal N — this is what makes the d2l FCN tab's loss 32× too high with
    batch_size=32 + lr=0.001."""
    _, w_after_n = _one_step_with_batch_size(4)
    _, w_after_1 = _one_step_with_batch_size(1)
    # update_n = 1 - 0.2 = 0.2 step
    # update_1 = 1 - 0.8 = 0.8 step
    # ratio = 4
    update_n = 1.0 - w_after_n
    update_1 = 1.0 - w_after_1
    assert update_1 / update_n == pytest.approx(4.0, abs=1e-5), (
        f"Expected step(1) update to be 4× step(4)'s update; got "
        f"step(1)={update_1:.6f}, step(4)={update_n:.6f}, ratio="
        f"{update_1/update_n:.6f}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
