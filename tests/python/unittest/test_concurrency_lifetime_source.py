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


def test_dnnl_concat_backward_owns_submemory_wrapper():
    contents = _read("src/operator/nn/dnnl/dnnl_concat.cc")
    body = contents.split("void DNNLConcatBackward", 1)[1].split("DNNLStream::Get()->Submit", 1)[0]

    assert "new dnnl::memory" not in body
    assert "dnnl::memory from_mem(from_md, gradz_mem->get_engine(), gradz_mem->get_data_handle())" in body
    assert "{{DNNL_ARG_FROM, from_mem}, {DNNL_ARG_TO, *gradi_mem.second}}" in body
    assert "dnnl::reorder(from_mem, *gradi_mem.second)" in body


def test_dnnl_rnn_backward_guards_null_state_cell_commit():
    contents = _read("src/operator/nn/dnnl/dnnl_rnn.cc")
    body = contents.split("void DNNLRnnOp::Backward", 1)[1].split("// Commit weights diff", 1)[0]

    assert "req[rnn_enum::kStateCell] != kNullOp" in body
    assert "CommitOutput(outputs[rnn_enum::kStateCell], diff_statecell_mem)" in body


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


def test_python_custom_prop_keeps_returned_pointer_buffers_alive():
    operator_py = _read("python/mxnet/operator.py")

    assert "shape_buffers = []" in operator_py
    assert "infer_shape_entry._ref_holder = [tensor_shapes, shape_buffers]" in operator_py
    assert "list_outputs_entry._ref_holder = [out, ret]" in operator_py
    assert "list_arguments_entry._ref_holder = [out, ret]" in operator_py
    assert "list_auxiliary_states_entry._ref_holder = [out, ret]" in operator_py
    assert "dep_buffer = c_array_buf(c_int, array('i', rdeps))" in operator_py
    assert "declare_backward_dependency_entry._ref_holder = [deps, dep_buffer]" in operator_py


def test_ctypes_ffi_keeps_global_handles_and_string_args_alive():
    contents = _read("python/mxnet/_ffi/_ctypes/function.py")

    assert "return _make_packed_func(handle, True)" in contents
    assert "cstr = c_str(arg)" in contents
    assert "temp_args.append(cstr)" in contents


def test_prefetching_iter_propagates_worker_errors_without_deadlock():
    contents = _read("python/mxnet/io/io.py")

    assert "self._prefetch_exceptions = [None for i in range(self.n_iter)]" in contents
    assert "except Exception as err" in contents
    assert "self._prefetch_exceptions[i] = err" in contents
    assert "self.data_ready[i].set()" in contents
    assert "def _check_prefetch_errors(self):" in contents
    assert "thread.join(timeout=5)" in contents


def test_quantize_asym_saturates_and_honors_output_requests():
    native = _read("src/operator/quantization/quantize_asym-inl.h")
    dnnl = _read("src/operator/quantization/dnnl/dnnl_quantize_asym-inl.h")

    assert "Min(Max(rounded, 0.0f)" in native
    assert "KERNEL_ASSIGN(out[i], req, quantized)" in native
    assert "AssignQuantizedRangeOutput<xpu>(s, outputs[1], scale, req[1])" in native
    assert "req[0] == kAddTo" in dnnl
    assert "KERNEL_ASSIGN(output_ptr[i], req[0], input_ptr[i])" in dnnl


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


def test_python_callback_ndarray_handles_are_python_owned():
    custom_cc = _read("src/operator/custom/custom.cc")
    c_api_function_cc = _read("src/c_api/c_api_function.cc")
    ctypes_ndarray = _read("python/mxnet/_ctypes/ndarray.py")
    c_api_cc = _read("src/c_api/c_api.cc")

    assert "check_call(_LIB.MXNDArrayFree(self.handle))" in ctypes_ndarray
    assert "delete static_cast<NDArray*>(handle)" in c_api_cc

    assert "std::unique_ptr<NDArray>" not in custom_cc
    assert "std::unique_ptr<NDArray>" not in c_api_function_cc


def test_numpy_binary_onednn_fallback_uses_layout_safe_fallback_compute():
    contents = _read("src/operator/numpy/np_elemwise_broadcast_op.h")
    body = contents.split("void NumpyBinaryOperatorComputeExCPU", 1)[1].split("#endif  // MXNET_USE_ONEDNN", 1)[0]

    assert "FallBackCompute(NumpyBinaryOperatorFallbackCPU<OP>, attrs, ctx, inputs, req, outputs)" in body
    assert "inputs[0].data()" not in body
    assert "outputs[0].data()" not in body


def test_tests_do_not_leak_print_options_or_kvstore_env():
    test_utils = _read("python/mxnet/test_utils.py")
    test_operator = _read("tests/python/unittest/test_operator.py")
    test_trainer = _read("tests/python/unittest/test_gluon_trainer.py")

    assert "np.set_printoptions" not in test_utils
    assert "np.set_printoptions" not in test_operator
    assert "with np.printoptions" in test_utils
    assert "os.putenv('MXNET_UPDATE_ON_KVSTORE'" not in test_trainer
    assert "previous_update_on_kvstore = os.environ.get('MXNET_UPDATE_ON_KVSTORE')" in test_trainer
    assert "os.environ.pop('MXNET_UPDATE_ON_KVSTORE', None)" in test_trainer


def test_dlpack_error_paths_release_owned_resources():
    contents = _read("python/mxnet/dlpack.py")

    assert "class _NumpyDLPackManager:" in contents
    assert "self.shape = (ctypes.c_int64 * array.ndim)(*array.shape)" in contents
    assert "manager.array.flags['WRITEABLE'] = manager.was_writeable" in contents
    assert "c_obj.manager_ctx = _make_manager_ctx(manager)" in contents
    assert "was_writeable = ndarray.flags['WRITEABLE']" in contents
    assert "dl_managed_tensor_deleter(ctypes.byref(c_obj))" in contents
    assert "ndarray.flags['WRITEABLE'] = was_writeable" in contents
    assert "check_call(_LIB.MXNDArrayFree(handle))" in contents
