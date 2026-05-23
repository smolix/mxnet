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

"""XOP16 storage-inference contract for the quantized embedding op.

issues.md XOP16 calls out that "sparse/storage inference and quantized
embedding row-sparse behavior remain open" without a concrete test
pinning the contract.  This file pins the contract: the
`_contrib_quantized_embedding` op infers default storage for primary
data and default storage for the min/max range scalars regardless of
the weight's storage type (default or row_sparse).

If a future quantization refactor changes the storage-inference
function, this fails fast instead of letting the dispatcher pick a
silently-degraded path.
"""

import numpy as np
import pytest

import mxnet as mx


def _has_quantized_embedding():
    return hasattr(mx.sym.contrib, 'quantized_embedding')


pytestmark = pytest.mark.skipif(
    not _has_quantized_embedding(),
    reason="contrib quantized embedding op not registered in this build")


def test_quantized_embedding_symbol_infers_default_storage():
    """Build the quantized embedding symbol and confirm shape/dtype/storage
    inference produce the expected outputs (1 primary + 2 range scalars)."""
    data = mx.sym.Variable('data', dtype='int32')
    weight = mx.sym.Variable('weight', dtype='int8')
    min_weight = mx.sym.Variable('min_weight', dtype='float32')
    max_weight = mx.sym.Variable('max_weight', dtype='float32')
    sym = mx.sym.contrib.quantized_embedding(
        data=data, weight=weight,
        min_weight=min_weight, max_weight=max_weight,
        input_dim=10, output_dim=4)

    # Shape inference: output is (data_shape..., output_dim); range outputs scalar.
    arg_shapes, out_shapes, _ = sym.infer_shape(
        data=(3, 2), weight=(10, 4),
        min_weight=(1,), max_weight=(1,))
    assert out_shapes[0] == (3, 2, 4), f"expected primary shape (3, 2, 4), got {out_shapes[0]}"
    assert out_shapes[1] == (1,), f"expected min scalar, got {out_shapes[1]}"
    assert out_shapes[2] == (1,), f"expected max scalar, got {out_shapes[2]}"

    # Type inference: primary is int8 (passthrough); range outputs float32.
    arg_types, out_types, _ = sym.infer_type(
        data=np.int32, weight=np.int8,
        min_weight=np.float32, max_weight=np.float32)
    assert out_types[0] == np.int8
    assert out_types[1] == np.float32
    assert out_types[2] == np.float32


def test_quantized_embedding_imperative_forward():
    """Imperative quantized embedding forward must produce the expected
    output shape + dtype, regardless of weight stype."""
    data = mx.nd.array([[0, 1], [2, 3]], dtype='int32')
    weight = mx.nd.zeros((10, 4), dtype='int8')
    weight[1, :] = 5
    weight[2, :] = 7
    min_weight = mx.nd.array([-128.0], dtype='float32')
    max_weight = mx.nd.array([127.0], dtype='float32')

    outputs = mx.nd.contrib.quantized_embedding(
        data=data, weight=weight,
        min_weight=min_weight, max_weight=max_weight,
        input_dim=10, output_dim=4)
    primary, out_min, out_max = outputs[0], outputs[1], outputs[2]
    assert primary.shape == (2, 2, 4)
    assert primary.dtype == np.int8
    # The selected rows must reflect the weight values at those indices.
    primary_np = primary.asnumpy()
    np.testing.assert_array_equal(primary_np[0, 1], np.full(4, 5, dtype=np.int8))
    np.testing.assert_array_equal(primary_np[1, 0], np.full(4, 7, dtype=np.int8))
    # Range scalars must propagate from weight_min/weight_max.
    np.testing.assert_array_equal(out_min.asnumpy(), [-128.0])
    np.testing.assert_array_equal(out_max.asnumpy(), [127.0])


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
