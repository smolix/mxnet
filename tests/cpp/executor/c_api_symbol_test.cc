/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <gtest/gtest.h>
#include <mxnet/c_api.h>
#include <mxnet/c_api_test.h>

#include <cstring>
#include <string>
#include <vector>

namespace {

void AssertMXSuccess(int status) {
  ASSERT_EQ(status, 0) << MXGetLastError();
}

AtomicSymbolCreator FindAtomicSymbolCreator(const char* op_name) {
  uint32_t size = 0;
  AtomicSymbolCreator* creators = nullptr;
  AssertMXSuccess(MXSymbolListAtomicSymbolCreators(&size, &creators));
  for (uint32_t i = 0; i < size; ++i) {
    const char* name = nullptr;
    AssertMXSuccess(MXSymbolGetAtomicSymbolName(creators[i], &name));
    if (std::strcmp(name, op_name) == 0) {
      return creators[i];
    }
  }
  return nullptr;
}

SymbolHandle CreateRepeatedInputAddSymbol() {
  SymbolHandle x = nullptr;
  SymbolHandle add = nullptr;
  AssertMXSuccess(MXSymbolCreateVariable("x", &x));

  AtomicSymbolCreator add_creator = FindAtomicSymbolCreator("elemwise_add");
  EXPECT_NE(add_creator, nullptr);
  if (add_creator == nullptr) {
    MXSymbolFree(x);
    return nullptr;
  }

  AssertMXSuccess(MXSymbolCreateAtomicSymbol(add_creator, 0, nullptr, nullptr, &add));
  SymbolHandle args[] = {x, x};
  AssertMXSuccess(MXSymbolCompose(add, "sum", 2, nullptr, args));
  MXSymbolFree(x);
  return add;
}

std::vector<std::string> ListOutputNames(SymbolHandle symbol) {
  uint32_t size = 0;
  const char** names = nullptr;
  AssertMXSuccess(MXSymbolListOutputs(symbol, &size, &names));

  std::vector<std::string> result;
  result.reserve(size);
  for (uint32_t i = 0; i < size; ++i) {
    result.emplace_back(names[i]);
  }
  return result;
}

void AssertVectorShape(const int* ndim,
                       const int** shape_data,
                       uint32_t index,
                       const std::vector<int>& expected) {
  ASSERT_EQ(ndim[index], static_cast<int>(expected.size()));
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(shape_data[index][i], expected[i]);
  }
}

}  // namespace

TEST(CAPISymbol, GetInputSymbolsIncludesBareVariable) {
  SymbolHandle x = nullptr;
  AssertMXSuccess(MXSymbolCreateVariable("x", &x));

  int input_size = 0;
  SymbolHandle* inputs = nullptr;
  AssertMXSuccess(MXSymbolGetInputSymbols(x, &inputs, &input_size));

  ASSERT_EQ(input_size, 1);
  EXPECT_EQ(ListOutputNames(inputs[0]), std::vector<std::string>({"x"}));

  for (int i = 0; i < input_size; ++i) {
    MXSymbolFree(inputs[i]);
  }
  MXSymbolFree(x);
}

TEST(CAPISymbol, GetInputSymbolsDeduplicatesRepeatedVariableInput) {
  SymbolHandle add = CreateRepeatedInputAddSymbol();
  ASSERT_NE(add, nullptr);

  uint32_t arg_size = 0;
  const char** arg_names = nullptr;
  AssertMXSuccess(MXSymbolListArguments(add, &arg_size, &arg_names));
  ASSERT_EQ(arg_size, 1U);
  EXPECT_STREQ(arg_names[0], "x");

  int input_size = 0;
  SymbolHandle* inputs = nullptr;
  AssertMXSuccess(MXSymbolGetInputSymbols(add, &inputs, &input_size));

  ASSERT_EQ(input_size, 1);
  EXPECT_EQ(ListOutputNames(inputs[0]), std::vector<std::string>({"x"}));

  for (int i = 0; i < input_size; ++i) {
    MXSymbolFree(inputs[i]);
  }
  MXSymbolFree(add);
}

TEST(CAPISymbol, DeletedSubgraphPreservesBoundaryInputsForInferenceAPIs) {
  SymbolHandle add = CreateRepeatedInputAddSymbol();
  ASSERT_NE(add, nullptr);

  const char* op_names[] = {"elemwise_add"};
  SymbolHandle partitioned = nullptr;
  AssertMXSuccess(MXBuildSubgraphByOpNames(add, "default", 1, op_names, &partitioned));

  uint32_t arg_size = 0;
  const char** arg_names = nullptr;
  AssertMXSuccess(MXSymbolListArguments(partitioned, &arg_size, &arg_names));
  ASSERT_EQ(arg_size, 1U);
  EXPECT_STREQ(arg_names[0], "x");

  SymbolHandle inputs = nullptr;
  AssertMXSuccess(MXSymbolGetInputs(partitioned, &inputs));
  EXPECT_EQ(ListOutputNames(inputs), std::vector<std::string>({"x"}));

  SymbolHandle children = nullptr;
  AssertMXSuccess(MXSymbolGetChildren(partitioned, &children));
  EXPECT_EQ(ListOutputNames(children), std::vector<std::string>({"x"}));

  const char* shape_keys[] = {"x"};
  const uint32_t shape_indptr[] = {0, 1};
  const int shape_data[] = {2};
  uint32_t in_shape_size = 0;
  uint32_t out_shape_size = 0;
  uint32_t aux_shape_size = 0;
  const int* in_shape_ndim = nullptr;
  const int* out_shape_ndim = nullptr;
  const int* aux_shape_ndim = nullptr;
  const int** in_shape_data = nullptr;
  const int** out_shape_data = nullptr;
  const int** aux_shape_data = nullptr;
  int complete = 0;
  AssertMXSuccess(MXSymbolInferShape(partitioned,
                                     1,
                                     shape_keys,
                                     shape_indptr,
                                     shape_data,
                                     &in_shape_size,
                                     &in_shape_ndim,
                                     &in_shape_data,
                                     &out_shape_size,
                                     &out_shape_ndim,
                                     &out_shape_data,
                                     &aux_shape_size,
                                     &aux_shape_ndim,
                                     &aux_shape_data,
                                     &complete));
  EXPECT_EQ(complete, 1);
  ASSERT_EQ(in_shape_size, 1U);
  ASSERT_EQ(out_shape_size, 1U);
  ASSERT_EQ(aux_shape_size, 0U);
  AssertVectorShape(in_shape_ndim, in_shape_data, 0, {2});
  AssertVectorShape(out_shape_ndim, out_shape_data, 0, {2});

  MXSymbolFree(children);
  MXSymbolFree(inputs);
  MXSymbolFree(partitioned);
  MXSymbolFree(add);
}

TEST(CAPISymbol, CutSubgraphDeduplicatesRepeatedBoundaryInput) {
  SymbolHandle add = CreateRepeatedInputAddSymbol();
  ASSERT_NE(add, nullptr);
  AssertMXSuccess(MXSymbolSetAttr(add, "__subgraph_name__", "sg"));

  int cut_input_size = 0;
  SymbolHandle* cut_inputs = nullptr;
  AssertMXSuccess(MXSymbolCutSubgraph(add, &cut_inputs, &cut_input_size));
  std::vector<SymbolHandle> cut_handles(cut_inputs, cut_inputs + cut_input_size);

  int graph_input_size = 0;
  SymbolHandle* graph_inputs = nullptr;
  AssertMXSuccess(MXSymbolGetInputSymbols(add, &graph_inputs, &graph_input_size));
  std::vector<SymbolHandle> graph_handles(graph_inputs, graph_inputs + graph_input_size);

  ASSERT_EQ(graph_input_size, 1);
  EXPECT_EQ(ListOutputNames(graph_handles[0]), std::vector<std::string>({"x"}));

  ASSERT_EQ(cut_input_size, graph_input_size);
  EXPECT_EQ(ListOutputNames(cut_handles[0]), ListOutputNames(graph_handles[0]));

  for (SymbolHandle input : graph_handles) {
    MXSymbolFree(input);
  }
  for (SymbolHandle input : cut_handles) {
    MXSymbolFree(input);
  }
  MXSymbolFree(add);
}
