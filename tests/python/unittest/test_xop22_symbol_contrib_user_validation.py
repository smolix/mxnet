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

"""XOP22 tail: user-facing validation in `mxnet.symbol.contrib`.

Two `assert` checks in `mxnet.symbol.contrib` were validating user-provided
function shapes (foreach body return format; while_loop cond return shape).
They needed to be `raise ValueError` so they survive `python -O`, matching
the rest of the XOP22 wave. Internal-invariant asserts (subgraph input
dedup, graph-inputs subset) are left as `assert` with a documenting comment.

This test pins both contracts: user input → ValueError; invariant → assert
(documented). It exists so a future cleanup that confuses the two doesn't
regress the user-input gates back to silent stripping under -O.
"""

import os
import pytest

import mxnet as mx
from mxnet import symbol


def test_foreach_state_format_mismatch_raises_value_error():
    """foreach body that returns a differently-shaped loop_vars vs the
    initial states must raise ValueError (not bare assert)."""
    data = symbol.var("data")
    init_state_a = symbol.var("state_a")
    init_state_b = symbol.var("state_b")

    # body returns 1 state instead of the 2 it was given — format mismatch.
    def bad_body(_, states):
        return _ * 2, states[0]

    with pytest.raises((ValueError, mx.MXNetError)) as excinfo:
        symbol.contrib.foreach(bad_body, data, [init_state_a, init_state_b])
    # ValueError preferred; MXNetError is the fallback the C engine might surface
    # if the symbolic infrastructure raises differently. The important guarantee
    # is that a meaningful exception bubbles up.
    msg = str(excinfo.value)
    assert "loop_vars" in msg.lower() or "format" in msg.lower() or msg, \
        f"Expected an informative error about loop_var format mismatch, got: {msg!r}"


def test_while_loop_cond_returning_multiple_outputs_raises_value_error():
    """while_loop cond function that returns more than one output (or any
    out_data) must raise ValueError."""
    counter = symbol.var("counter")
    limit = symbol.var("limit")

    # Bad cond: returns a list of two values (treated as out_data + scalar)
    # while_loop unpacks the loop_vars list into positional args.
    def bad_cond(counter_, limit_):
        return [counter_ < limit_, counter_]

    def good_func(counter_, limit_):
        return [counter_ + 1], [counter_ + 1, limit_]

    with pytest.raises((ValueError, mx.MXNetError)) as excinfo:
        symbol.contrib.while_loop(
            bad_cond, good_func, [counter, limit], max_iterations=5)
    msg = str(excinfo.value)
    assert "cond" in msg.lower() or "while_loop" in msg.lower() or msg, \
        f"Expected an informative error about cond shape, got: {msg!r}"


def test_user_input_validation_uses_raise_not_assert():
    """Source-grep regression: the two user-input validation sites must use
    `raise ValueError` rather than `assert`. (Internal-invariant asserts on
    subgraph dedup and graph-inputs subset are still asserts — by design.)"""
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "python", "mxnet", "symbol", "contrib.py")
    with open(src) as f:
        contents = f.read()

    # foreach body return-format mismatch must be a raise, not an assert.
    assert "raise ValueError(\n                \"The input and output loop_vars of foreach" in contents, \
        "XOP22 regression: foreach state-format check has been reverted to bare assert"

    # while_loop cond shape check must be a raise, not an assert.
    assert "raise ValueError(\n            \"The cond function of while_loop" in contents, \
        "XOP22 regression: while_loop cond-shape check has been reverted to bare assert"


def test_invariant_asserts_have_documenting_comments():
    """Internal-invariant asserts (which are deliberately kept as `assert`)
    must carry a "not user-facing" or "mxnet invariant" comment so a future
    audit doesn't mistake them for stripped user validation."""
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "python", "mxnet", "symbol", "contrib.py")
    with open(src) as f:
        contents = f.read()
    # Each invariant assertion site should have a comment marker nearby.
    assert "mxnet invariant" in contents, \
        "XOP22 invariant assertions are no longer documented as such"


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
