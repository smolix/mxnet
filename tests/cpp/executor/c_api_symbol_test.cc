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
