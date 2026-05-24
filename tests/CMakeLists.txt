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

# ---[ Google Test
if(NOT GTEST_ROOT)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
    set(GTEST_FOUND ON)
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
    set(GTEST_FOUND ON)
  else()
    find_package(GTest)
  endif()
endif()

# FIXME MSVC unit test linking issue
if(GTEST_FOUND AND NOT MSVC)
  enable_testing()

  file(GLOB_RECURSE UNIT_TEST_HEADERS "cpp/*.h")

  include_directories(${GTEST_INCLUDE_DIR})
  include_directories(cpp/include)
  include_directories(${PROJECT_SOURCE_DIR}/cpp-package/include)

  if(NOT PRIVATE_RUNTIME_DIR)
    if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
      set(PRIVATE_RUNTIME_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    else()
      set(PRIVATE_RUNTIME_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()
  endif()

  set(MXNET_UNIT_TEST_COMMON_LIBS
    ${GTEST_LIBRARY}
    dmlc
    ${nnvm_LINKER_LIBS}
    ${mxnet_LINKER_LIBS}
    mxnet
    ${pslite_LINKER_LIBS}
    )
  set(MXNET_UNIT_TEST_MAIN cpp/test_main.cc)
  set(MXNET_UNIT_TEST_TARGETS)
  set(MXNET_UNIT_TEST_WRAPPER_ENTRIES)

  function(mxnet_add_unit_test_executable suffix default_filter)
    set(target ${PROJECT_NAME}_unit_tests_${suffix})
    add_executable(${target} ${MXNET_UNIT_TEST_MAIN} ${ARGN} ${UNIT_TEST_HEADERS})
    set_property(TARGET ${target}
                 PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PRIVATE_RUNTIME_DIR})
    if(NOT "${default_filter}" STREQUAL "")
      target_compile_definitions(${target} PRIVATE
        MXNET_UNIT_TEST_DEFAULT_FILTER="${default_filter}")
    endif()
    target_link_libraries(${target} ${MXNET_UNIT_TEST_COMMON_LIBS})
    add_test(NAME AllTestsIn${PROJECT_NAME}UnitTests_${suffix}
             COMMAND $<TARGET_FILE:${target}>)
    set(MXNET_UNIT_TEST_TARGETS ${MXNET_UNIT_TEST_TARGETS} ${target} PARENT_SCOPE)
    string(APPEND MXNET_UNIT_TEST_WRAPPER_ENTRIES
           "  \"${suffix}|${PRIVATE_RUNTIME_DIR}/${target}\"\n")
    set(MXNET_UNIT_TEST_WRAPPER_ENTRIES
        "${MXNET_UNIT_TEST_WRAPPER_ENTRIES}" PARENT_SCOPE)
  endfunction()

  mxnet_add_unit_test_executable(engine "" 
    cpp/engine/engine_shutdown_test.cc
    cpp/engine/omp_test.cc
    cpp/engine/thread_local_test.cc
    cpp/engine/threaded_engine_invariants_test.cc
    cpp/engine/threaded_engine_test.cc)

  mxnet_add_unit_test_executable(executor ""
    cpp/executor/c_api_cached_op_test.cc
    cpp/executor/c_api_symbol_test.cc
    cpp/executor/cpp_executor_test.cc)

  mxnet_add_unit_test_executable(kvstore ""
    cpp/kvstore/gpu_topology_test.cc)

  mxnet_add_unit_test_executable(misc_storage ""
    cpp/misc/base.cc
    cpp/misc/libinfo_test.cc
    cpp/storage/storage_test.cc)

  mxnet_add_unit_test_executable(operator_batchnorm ""
    cpp/operator/batchnorm_test.cc)

  mxnet_add_unit_test_executable(operator_dnnl_support ""
    cpp/operator/dnnl_test.cc)

  mxnet_add_unit_test_executable(operator_dnnl_basic
    "IMPERATIVE.CopyOp:IMPERATIVE.CopyBackwardsOp:IMPERATIVE.ActOp:IMPERATIVE.ActBackwardsOp:IMPERATIVE.SumOp:IMPERATIVE.SumBackwardsOp:IMPERATIVE.ConcatOp:IMPERATIVE.ConcatBackwardsOp:IMPERATIVE.LRNOp:IMPERATIVE.SoftmaxOp"
    cpp/operator/dnnl_operator_test.cc)

  mxnet_add_unit_test_executable(operator_dnnl_fc_pool
    "IMPERATIVE.FullyConnectedOp:IMPERATIVE.PoolingOp"
    cpp/operator/dnnl_operator_test.cc)

  mxnet_add_unit_test_executable(operator_dnnl_conv
    "IMPERATIVE.ConvOp"
    cpp/operator/dnnl_operator_test.cc)

  mxnet_add_unit_test_executable(operator_dnnl_deconv
    "IMPERATIVE.DeconvOp"
    cpp/operator/dnnl_operator_test.cc)

  mxnet_add_unit_test_executable(operator_dnnl_batchnorm
    "IMPERATIVE.BNOp:IMPERATIVE.NumpySumReduceDNNLAddToReq:IMPERATIVE.QuantizedBatchNormDNNLReq"
    cpp/operator/dnnl_operator_test.cc)

  mxnet_add_unit_test_executable(operator_perf ""
    cpp/operator/activation_perf.cc
    cpp/operator/coreop_perf.cc
    cpp/operator/dropout_perf.cc
    cpp/operator/fully_conn_perf.cc
    cpp/operator/slice_channel_perf.cc)

  mxnet_add_unit_test_executable(operator_runner ""
    cpp/operator/runner/core_op_runner_test.cc)

  mxnet_add_unit_test_executable(operator_tune ""
    cpp/operator/tune/operator_tune_test.cc)

  mxnet_add_unit_test_executable(operator_misc ""
    cpp/operator/image_crop_test.cc
    cpp/operator/krprod_test.cc
    cpp/operator/sequence_reverse_test.cc
    cpp/operator/topk_test.cc
    cpp/operator/unique_sample_test.cc)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_unit_test_shards.sh.in
                 ${PRIVATE_RUNTIME_DIR}/${PROJECT_NAME}_unit_tests
                 @ONLY)
  execute_process(COMMAND chmod +x ${PRIVATE_RUNTIME_DIR}/${PROJECT_NAME}_unit_tests)

  add_custom_target(${PROJECT_NAME}_unit_tests DEPENDS ${MXNET_UNIT_TEST_TARGETS})
else()
  message(STATUS "Google Test not found")
endif()
