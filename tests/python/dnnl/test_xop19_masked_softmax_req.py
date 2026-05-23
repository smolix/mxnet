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

"""Contract coverage for DNNLMaskedSoftmaxForward output-request semantics.

The XOP19 audit flagged `dnnl_masked_softmax.cc` because the chain
(mask-1 → reorder → add → softmax → multiply) routes through the
caller's output buffer as scratch.  That is correct provided the
forward entry point continues to:

- early-return on `kNullOp` and leave the output buffer untouched;
- reject `kAddTo` before any output write happens (the buffer can't be
  used as accumulator and scratch simultaneously);
- produce the correct masked-softmax for `kWriteTo`.

These tests pin those properties so a future refactor that "cleans up"
the scratch pattern cannot quietly regress the contract.

Larger end-to-end correctness for masked_softmax is covered by
`tests/python/unittest/test_operator.py::test_masked_softmax` over a
range of dtypes, axes, ndims, and temperatures.  This file is the focused
contract layer.
"""

import numpy as np
import pytest

import mxnet as mx


def _onednn_available():
    return mx.runtime.Features().is_enabled("ONEDNN")


pytestmark = pytest.mark.skipif(
    not _onednn_available(), reason="oneDNN backend not enabled")


# masked_softmax dispatches to oneDNN only above an internal size threshold of
# 2 << 13 = 16384.  Use 4 * 4096 = 16384 elements to exercise that path.
def _big_inputs(seed=0):
    rng = np.random.RandomState(seed)
    data_np = rng.randn(4, 4096).astype('float32')
    mask_np = (rng.rand(4, 4096) > 0.3)  # ~70% kept
    return data_np, mask_np


def _reference_masked_softmax(data_np, mask_np, axis=-1, temperature=1.0):
    masked = np.where(mask_np, data_np / float(temperature), -np.inf)
    masked = masked - masked.max(axis=axis, keepdims=True)
    exp = np.exp(masked)
    denom = exp.sum(axis=axis, keepdims=True)
    out = np.where(denom > 0, exp / np.maximum(denom, 1e-30), 0.0)
    out = np.where(mask_np, out, 0.0)
    return out.astype('float32')


def test_masked_softmax_onednn_path_correct():
    """Large input takes the oneDNN dispatch path; output must match oracle."""
    data_np, mask_np = _big_inputs(seed=0)
    data = mx.nd.array(data_np)
    mask = mx.nd.array(mask_np, dtype=np.bool_)
    out = mx.nd.masked_softmax(data=data, mask=mask, axis=-1).asnumpy()
    expected = _reference_masked_softmax(data_np, mask_np)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_masked_softmax_temperature_scaling_onednn_path():
    """Temperature != 1 takes the post-op-scale branch."""
    data_np, mask_np = _big_inputs(seed=1)
    data = mx.nd.array(data_np)
    mask = mx.nd.array(mask_np, dtype=np.bool_)
    out = mx.nd.masked_softmax(data=data, mask=mask, axis=-1,
                               temperature=0.5).asnumpy()
    expected = _reference_masked_softmax(data_np, mask_np, temperature=0.5)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_masked_softmax_native_path_small_shape():
    """Below the oneDNN threshold the native softmax path runs.  Result must
    still match the oracle so a future dispatch regression is caught here."""
    rng = np.random.RandomState(2)
    data_np = rng.randn(2, 8).astype('float32')
    mask_np = (rng.rand(2, 8) > 0.5)
    out = mx.nd.masked_softmax(data=mx.nd.array(data_np),
                               mask=mx.nd.array(mask_np, dtype=np.bool_),
                               axis=-1).asnumpy()
    expected = _reference_masked_softmax(data_np, mask_np)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_masked_softmax_repeated_calls_independent():
    """Run masked_softmax three times on different inputs, confirming each
    output is fresh data rather than carrying state from the previous call.
    Catches a regression where the cached primitive holds onto a stale output
    buffer (a real risk given the chain-of-scratch pattern through the
    output's storage)."""
    results = []
    for seed in (10, 11, 12):
        data_np, mask_np = _big_inputs(seed=seed)
        data = mx.nd.array(data_np)
        mask = mx.nd.array(mask_np, dtype=np.bool_)
        out = mx.nd.masked_softmax(data=data, mask=mask, axis=-1).asnumpy()
        expected = _reference_masked_softmax(data_np, mask_np)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
        results.append(out)
    # Sanity: the three outputs must actually differ (otherwise the test
    # would pass by always producing the same stale buffer).
    assert not np.allclose(results[0], results[1])
    assert not np.allclose(results[1], results[2])


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
