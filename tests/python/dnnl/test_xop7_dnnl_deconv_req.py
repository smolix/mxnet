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

"""Coverage for the XOP7 leftover: DNNL Deconvolution weight-grad fast path.

`DNNLDeconvBwd::WeightsGradMem` used to take an unconditional fast path on
`req == kWriteTo` that called `CreateDNNLData` and stuffed the result into a
`{OutDataOp::Noop, mem}` pair.  Two problems:

1. `CreateDNNLData` returns nullptr for views or storage that can't be bound
   to the swapped descriptor; the resulting nullptr would later trip
   `CommitOutput` / primitive arg lookup.

2. The fall-through `CreateDNNLWeightGrad` helper did not honor `kNullOp`,
   so a caller that explicitly opted out of receiving weight gradients
   still saw their buffer written.

Both are fixed: the fast path falls back to `CreateDNNLWeightGrad` when the
direct bind returns nullptr, and `CreateDNNLWeightGrad` now allocates a
throwaway tmp + `Noop` for `kNullOp`.

These tests pin the user-visible behavior: `grad_req='null'` leaves the
sentinel untouched, `grad_req='add'` accumulates, `grad_req='write'`
overwrites.
"""

import numpy as np
import pytest

import mxnet as mx


def _onednn_available():
    feats = mx.runtime.Features()
    return feats.is_enabled("ONEDNN")


pytestmark = pytest.mark.skipif(
    not _onednn_available(), reason="oneDNN backend not enabled")


def _run_deconv_backward(weight_grad_req, weight_grad_init):
    """Run a tiny Deconvolution1D forward+backward via Symbol._bind with a
    pre-initialized weight gradient buffer regardless of grad_req so we can
    observe whether the buffer was written.  Returns the weight grad buffer
    after backward."""
    np.random.seed(0)
    data_np = np.random.randn(2, 3, 4).astype(np.float32)
    weight_np = np.random.randn(3, 6, 3).astype(np.float32)  # I, O, K
    data_sym = mx.sym.Variable('data')
    weight_sym = mx.sym.Variable('weight')
    out_sym = mx.sym.Deconvolution(
        data=data_sym, weight=weight_sym,
        kernel=(3,), stride=(1,), pad=(0,),
        num_filter=6, no_bias=True)
    data_arr = mx.nd.array(data_np)
    weight_arr = mx.nd.array(weight_np)
    data_grad = mx.nd.zeros(data_arr.shape)
    weight_grad = mx.nd.array(weight_grad_init)
    exe = out_sym._bind(
        ctx=mx.cpu(),
        args={'data': data_arr, 'weight': weight_arr},
        args_grad={'data': data_grad, 'weight': weight_grad},
        grad_req={'data': 'write', 'weight': weight_grad_req})
    exe.forward(is_train=True)
    exe.backward(mx.nd.ones_like(exe.outputs[0]))
    mx.nd.waitall()
    return weight_grad.asnumpy()


def test_deconv_weight_grad_null_preserves_sentinel():
    sentinel = 17.0
    init = np.full((3, 6, 3), sentinel, dtype=np.float32)
    out = _run_deconv_backward('null', init)
    # kNullOp must leave the buffer untouched.
    np.testing.assert_array_equal(out, init)


def test_deconv_weight_grad_write_overwrites():
    init = np.full((3, 6, 3), 99.0, dtype=np.float32)
    out = _run_deconv_backward('write', init)
    # kWriteTo must overwrite the sentinel.  The buffer should no longer
    # carry the 99s; the actual gradient depends on the random input but
    # is bounded.
    assert not np.allclose(out, 99.0)


def test_deconv_weight_grad_add_accumulates():
    init = np.full((3, 6, 3), 17.0, dtype=np.float32)
    written = _run_deconv_backward('add', init)
    overwritten = _run_deconv_backward('write', np.zeros_like(init))
    # add result must equal write result + sentinel within fp32 tolerance.
    np.testing.assert_allclose(written, overwritten + 17.0, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
