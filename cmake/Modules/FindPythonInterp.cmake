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

# Compatibility shim for vendored dependencies that still call the removed
# FindPythonInterp module under CMake 4. Prefer CMake's modern Python3 package
# while preserving the legacy result variables those dependencies consume.

if(PYTHON_EXECUTABLE AND NOT Python3_EXECUTABLE)
  set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

find_package(Python3 QUIET COMPONENTS Interpreter)

set(PYTHONINTERP_FOUND ${Python3_Interpreter_FOUND})
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Path to Python interpreter")
set(PYTHON_VERSION_STRING "${Python3_VERSION}")
set(PYTHON_VERSION_MAJOR "${Python3_VERSION_MAJOR}")
set(PYTHON_VERSION_MINOR "${Python3_VERSION_MINOR}")
set(PYTHON_VERSION_PATCH "${Python3_VERSION_PATCH}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PythonInterp
  REQUIRED_VARS PYTHON_EXECUTABLE
  VERSION_VAR PYTHON_VERSION_STRING)
