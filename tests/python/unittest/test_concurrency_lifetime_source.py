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

from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


def _read(rel):
    return (REPO / rel).read_text()


def test_legacy_ndarray_op_owns_callback_wrappers():
    contents = _read("src/operator/custom/ndarray_op.cc")

    assert "std::vector<std::unique_ptr<NDArray>> nd_wrappers" in contents
    assert "new NDArray(blob, ndctx.dev_id)" in contents
    assert "reinterpret_cast<void*>(new NDArray" not in contents


def test_threaded_engine_wait_ops_run_during_shutdown():
    header = _read("src/engine/threaded_engine.h")
    source = _read("src/engine/threaded_engine.cc")

    assert "!shutdown_phase_ || threaded_opr->wait" in header
    assert "std::make_shared<std::atomic<bool>>(false)" in source
    assert "[this, &done]" not in source


def test_nccl_kvstore_drains_engine_before_teardown():
    contents = _read("src/kvstore/kvstore_nccl.h")

    assert "Engine::Get()->WaitForAll();" in contents


def test_dnnl_fallback_does_not_invalidate_null_outputs():
    contents = _read("src/imperative/imperative_utils.h")

    invalidate_body = contents.split("void InvalidateOutputs", 1)[1].split("}", 1)[0]
    assert "reqs[i] == kWriteTo" in invalidate_body
    assert "kNullOp" not in invalidate_body


def test_dnnl_activation_backward_uses_commit_output_path():
    contents = _read("src/operator/nn/dnnl/dnnl_act.cc")
    body = contents.split("void DNNLActivationBackward", 1)[1].split("void DNNLLeakyReluBackward", 1)[0]

    assert "CreateDNNLMem(in_grad, bwd.bwd_pd.diff_src_desc(), req[0])" in body
    assert "CommitOutput(in_grad, diff_src_memory)" in body
    assert "CreateDNNLData" not in body


def test_dnnl_batch_norm_forward_honors_output_requests():
    contents = _read("src/operator/nn/dnnl/dnnl_batch_norm.cc")
    body = contents.split("void DNNLBNForward::Execute", 1)[1].split("// v3: build", 1)[0]

    assert "CreateDNNLMem(out, fwd_dst_desc, req[batchnorm::kOut], &data)" in body
    assert "CommitOutput(out, out_mem)" in body
    assert "KERNEL_ASSIGN(omean[i], req[batchnorm::kMean]" in body
    assert "KERNEL_ASSIGN(ovar[i], req[batchnorm::kVar]" in body
    assert "NDArray saved_mean(outMean.shape(), outMean.ctx(), false, outMean.dtype())" in body
    assert "NDArray saved_var(outVar.shape(), outVar.ctx(), false, outVar.dtype())" in body
    assert "KERNEL_ASSIGN(out_mean_ptr[i], req[batchnorm::kMean]" in body
    assert "KERNEL_ASSIGN(out_var_ptr[i]," in body


def test_python_custom_callbacks_keep_ctypes_arrays_alive():
    operator_py = _read("python/mxnet/operator.py")
    autograd_py = _read("python/mxnet/autograd.py")

    assert "callback_array = c_array(CFUNCTYPE(c_int), callbacks)" in operator_py
    assert "context_array = c_array(c_void_p, contexts)" in operator_py
    assert "op._ref_holder = [ret, callbacks, callback_array, context_array]" in operator_py
    assert "op_prop._ref_holder = [ret, callbacks, callback_array, context_array]" in operator_py
    assert "callback_array = c_array(CFUNCTYPE(c_int), callbacks)" in autograd_py
    assert "context_array = c_array(c_void_p, [None]*len(callbacks))" in autograd_py
    assert "Function._registry.ref_holder[key] = (context, callbacks, callback_array, context_array)" in autograd_py


def test_custom_create_operator_releases_callback_list_only_after_success():
    contents = _read("src/operator/custom/custom.cc")
    body = contents.split("OpStatePtr CreateState", 1)[1].split("void ForwardEx", 1)[0]

    assert "std::unique_ptr<MXCallbackList> op_info(new MXCallbackList)" in body
    assert "op_info.get()" in body
    assert "op_info.release()" in body


def test_packed_ret_value_owns_copied_ndarray_handles_until_handoff():
    contents = _read("include/mxnet/runtime/packed_func.h")
    body = contents.split("class MXNetRetValue", 1)[1].split("inline DLDataType String2DLDataType", 1)[0]

    assert "bool ndarray_handle_is_owned_{false}" in body
    assert "ndarray_handle_is_owned_   = true" in body
    assert "delete ptr<NDArray>()" in body
    assert "ndarray_handle_is_owned_   = false" in body
