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

"""d2l-mxnet-issues.md Issue 6 — lr_scheduler ``epoch_size`` convenience.

Cross-framework measurement confirmed all three scheduler ecosystems
(MXNet, PyTorch, JAX/Optax) count caller-supplied ``step`` calls and have
no built-in epoch concept.  The d2l-mxnet ``lr-scheduler`` notebook fed
epoch-scale milestones into the Gluon scheduler whose counter is bumped
per-minibatch by ``Trainer.step()``, producing the ~2× higher train loss
vs the PyTorch tab — which uses the same milestones with a per-epoch
``scheduler.step()`` call in its training loop.

The fix is a new ``epoch_size`` kw added to ``FactorScheduler``,
``MultiFactorScheduler``, ``PolyScheduler``, and ``CosineScheduler``: when
set, the scheduler multiplies its ``step``/``max_update`` arguments by
``epoch_size`` internally, so callers can pass epoch indices and get
PyTorch-equivalent behaviour under Gluon's per-minibatch counter.

These tests pin:
1. The default (no ``epoch_size``) behaviour is unchanged.
2. With ``epoch_size``, milestones fire at the expected per-minibatch step
   = epoch * num_batches.
3. Invalid ``epoch_size`` values are rejected with a clear ValueError.
"""

import pytest

from mxnet.lr_scheduler import (
    CosineScheduler,
    FactorScheduler,
    MultiFactorScheduler,
    PolyScheduler,
)


# -----------------------------------------------------------------------
# MultiFactorScheduler — the most common d2l case
# -----------------------------------------------------------------------

def test_multifactor_default_step_semantics_unchanged():
    """Without epoch_size, step=[15, 30] continues to mean update counts."""
    s = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
    assert s(0) == 0.5
    assert s(15) == 0.5  # boundary not yet crossed (uses `>` semantics)
    assert s(16) == 0.25
    assert s(30) == 0.25
    assert s(31) == 0.125


def test_multifactor_with_epoch_size_drops_at_epoch_boundary():
    """With epoch_size=200 and step=[15, 30], LR drops at update 15*200=3000."""
    s = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5,
                              epoch_size=200)
    # Plenty of in-epoch updates → unchanged
    assert s(0) == 0.5
    assert s(2999) == 0.5
    # 3000 is the boundary, drop fires at 3001 (consistent with existing `>` semantics)
    assert s(3000) == 0.5
    assert s(3001) == 0.25
    assert s(6000) == 0.25
    assert s(6001) == 0.125


def test_multifactor_invalid_epoch_size_rejected():
    with pytest.raises(ValueError, match="epoch_size"):
        MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5, epoch_size=0)
    with pytest.raises(ValueError, match="epoch_size"):
        MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5, epoch_size=-3)
    with pytest.raises(ValueError, match="epoch_size"):
        MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5, epoch_size="bad")


# -----------------------------------------------------------------------
# FactorScheduler
# -----------------------------------------------------------------------

def test_factor_default_step_semantics_unchanged():
    s = FactorScheduler(step=10, factor=0.5, base_lr=1.0)
    assert s(0) == 1.0
    assert s(10) == 1.0
    assert s(11) == 0.5
    assert s(21) == 0.25


def test_factor_with_epoch_size_scales_step():
    s = FactorScheduler(step=10, factor=0.5, base_lr=1.0, epoch_size=50)
    # Boundary at update step 10 * 50 = 500
    assert s(0) == 1.0
    assert s(500) == 1.0
    assert s(501) == 0.5


# -----------------------------------------------------------------------
# CosineScheduler — the d2l case where max_update=20 collapsed decay
# -----------------------------------------------------------------------

def test_cosine_default_max_update_unchanged():
    s = CosineScheduler(max_update=100, base_lr=1.0, final_lr=0.01)
    assert s(0) == pytest.approx(1.0)
    # Halfway through decay
    s_50 = s(50)
    assert 0.4 < s_50 < 0.6
    assert s(100) == pytest.approx(0.01, abs=1e-6)


def test_cosine_with_epoch_size_extends_decay():
    """The d2l book passes max_update=20 hoping for 20 epochs of cosine
    decay; pre-fix this collapsed to 20 minibatches.  epoch_size=200 makes
    it span the intended ~4000-update window."""
    s = CosineScheduler(max_update=20, base_lr=1.0, final_lr=0.01,
                         epoch_size=200)
    assert s(0) == pytest.approx(1.0)
    # Halfway through the 4000-update window
    s_2000 = s(2000)
    assert 0.4 < s_2000 < 0.6
    # End of decay window
    assert s(4000) == pytest.approx(0.01, abs=1e-6)
    # Past the end — pinned at final_lr (existing CosineScheduler behaviour)
    assert s(5000) == pytest.approx(0.01, abs=1e-6)


def test_cosine_invalid_epoch_size_rejected():
    with pytest.raises(ValueError, match="epoch_size"):
        CosineScheduler(max_update=20, base_lr=1.0, epoch_size=0)


# -----------------------------------------------------------------------
# PolyScheduler — symmetric coverage
# -----------------------------------------------------------------------

def test_poly_with_epoch_size_scales_max_update():
    s = PolyScheduler(max_update=10, base_lr=1.0, pwr=2, final_lr=0.0,
                       epoch_size=100)
    assert s(0) == pytest.approx(1.0)
    # At the end of the scaled window
    assert s(1000) == pytest.approx(0.0, abs=1e-6)


# -----------------------------------------------------------------------
# End-to-end: verify the d2l notebook fix shape
# -----------------------------------------------------------------------

def test_pytorch_equivalence_under_minibatch_counter():
    """Simulate a Gluon Trainer driven per-minibatch with 200 batches/epoch
    and epoch milestones [15, 30].  At the boundary epoch 15 (update
    3000) the LR should drop to 0.25 — matching PyTorch's MultiStepLR
    with the same milestones driven by a per-epoch scheduler.step()
    call."""
    num_batches = 200
    s = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5,
                              epoch_size=num_batches)
    # Walk through 35 epochs at minibatch granularity
    lrs_at_epoch_end = {}
    for epoch in range(35):
        last_update = (epoch + 1) * num_batches  # 200, 400, ..., 7000
        lrs_at_epoch_end[epoch] = s(last_update)
    # Before milestone 15 → base_lr
    for e in range(0, 15):
        assert lrs_at_epoch_end[e] == 0.5, (
            f"LR at end of epoch {e} should be 0.5, got {lrs_at_epoch_end[e]}")
    # After milestone 15 → 0.25
    for e in range(15, 30):
        assert lrs_at_epoch_end[e] == 0.25, (
            f"LR at end of epoch {e} should be 0.25, got {lrs_at_epoch_end[e]}")
    # After milestone 30 → 0.125
    for e in range(30, 35):
        assert lrs_at_epoch_end[e] == 0.125, (
            f"LR at end of epoch {e} should be 0.125, got {lrs_at_epoch_end[e]}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
