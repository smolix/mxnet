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

import re
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _api_sources():
    root = _repo_root() / "src" / "api"
    return sorted(root.rglob("*.cc")) + sorted(root.rglob("*.h"))


def test_api_output_adt_wrappers_do_not_drop_invoke_owned_ndarrays():
    """Guard the ownership rule for `Invoke(..., outputs=nullptr)` results.

    `Invoke` returns heap-allocated `NDArray*` output handles when the caller
    does not pass an output array.  Multi-output Python API wrappers must copy
    those values into runtime `NDArrayHandle` objects for ADT returns, and then
    delete the original heap handles.  The leaking pattern was:

        std::vector<NDArrayHandle> handles;
        handles.emplace_back(ndoutputs[i]);
        *ret = ADT(0, handles.begin(), handles.end());

    That copies the NDArray values but loses the original `NDArray*` handles.
    Large outputs, such as `_npx.rnn(..., state_outputs=True)`, then leak one
    output allocation per call.
    """
    banned_adt_wrapper = re.compile(
        r"std::vector\s*<\s*NDArrayHandle\s*>\s+(\w+)\s*;"
        r"(?s:.{0,1200}?)"
        r"ADT\s*\(\s*0\s*,\s*\1\.begin\s*\(\s*\)\s*,\s*\1\.end\s*\(\s*\)\s*\)"
    )
    banned_single_output_copy = re.compile(
        r"\*ret\s*=\s*NDArrayHandle\s*\(\s*ndoutputs\s*\[\s*0\s*\]\s*\)"
    )

    offenders = []
    for path in _api_sources():
        rel = path.relative_to(_repo_root())
        # The helper is the only legitimate place to copy outputs into
        # NDArrayHandle wrappers; it deletes the original heap handles.
        if rel.as_posix() == "src/api/operator/utils.cc":
            continue
        text = path.read_text()
        if banned_adt_wrapper.search(text):
            offenders.append(f"{rel}: copied Invoke outputs into ADT without ownership cleanup")
        if banned_single_output_copy.search(text):
            offenders.append(f"{rel}: copied single Invoke output instead of transferring handle")

    assert not offenders, "\n".join(offenders)


def test_operator_api_uses_shared_output_adt_ownership_helper():
    """The source tree should keep using the shared owning ADT helper.

    This is intentionally broad: the original leak pattern existed in many
    wrappers, not just `_npx.rnn`.
    """
    root = _repo_root() / "src" / "api" / "operator"
    helper_calls = []
    for path in sorted(root.rglob("*.cc")):
        if path.name == "utils.cc":
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if "CreateADTFromOutputVector(&ndoutputs" in line:
                helper_calls.append(f"{path.relative_to(_repo_root())}:{line_no}")

    assert len(helper_calls) >= 20, (
        "Expected multi-output operator wrappers to use "
        "CreateADTFromOutputVector; found only:\n" + "\n".join(helper_calls)
    )


def test_c_api_ndarray_implicit_outputs_are_raii_owned():
    """Implicit C API outputs must be cleaned up if invoke throws.

    `MXImperativeInvokeImpl` and `MXInvokeCachedOp` allocate output handles
    before the actual operator execution.  Those handles need frame-local
    ownership until successful return to avoid leaking on API_END exception
    paths.
    """
    text = (_repo_root() / "src" / "c_api" / "c_api_ndarray.cc").read_text()
    assert "std::vector<std::unique_ptr<NDArray>> owned_outputs" in text
    assert "owned_outputs[i].release()" in text
    assert "delete ndoutputs" not in text
