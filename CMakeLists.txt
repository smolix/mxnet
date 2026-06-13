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

cmake_minimum_required(VERSION 3.13)

# CMake 4 no longer accepts subprojects that request very old policy
# compatibility. Several vendored dependencies still declare older minima, so
# keep their policy floor explicit while preserving MXNet's existing CMake
# minimum for downstream builders.
set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING
    "Minimum CMake policy version used by vendored dependencies")

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW CACHE STRING
    "Let option() honor normal variables in vendored dependencies")
set(CMAKE_POLICY_DEFAULT_CMP0148 NEW CACHE STRING
    "Use MXNet's compatibility module for legacy FindPythonInterp calls")

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0144 NEW CACHE STRING
      "Use upper-case <PACKAGENAME>_ROOT variables in vendored dependencies")
endif()

# arm64 macOS is available from macOS 11 onward. Detect the host arch before
# project() initializes compiler flags; CMAKE_HOST_SYSTEM_PROCESSOR is not
# reliably populated yet on all CMake versions.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  execute_process(COMMAND uname -m OUTPUT_VARIABLE MXNET_HOST_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(MXNET_HOST_ARCH MATCHES "^(arm64|aarch64)$"
     AND (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "11.0"))
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
  endif()
endif()

# workaround to store CMAKE_CROSSCOMPILING because is getting reset by the project command
if(CMAKE_CROSSCOMPILING)
  set(__CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING})
  set(__CMAKE_CROSSCOMPILING_OVERRIDE ON)
endif()

project(mxnet C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # GNU extensions used by src/operator/random/shuffle_op.cc

# Sanity checks for some popular compilers. Make sure their version is
# sufficient. Cmake also automatically checks if a compiler supports c++17. But
# some compilers claim they support c++17 without actually implementing crucial
# parts of the standard leading to hard to understand compilation errors.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "MXNet 2 requires a C++17 compatible compiler. Please update to GCC version 7 or newer.")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    message(FATAL_ERROR "MXNet 2 requires a C++17 compatible compiler. Please update to Clang version 6 or newer.")
  endif()
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  # Load config.cmake only if mxnet is not compiled as a dependency of another project
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
endif()

if(__CMAKE_CROSSCOMPILING_OVERRIDE)
  set(CMAKE_CROSSCOMPILING ${__CMAKE_CROSSCOMPILING})
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Utils.cmake)

include(CMakeDependentOption)

# On macOS, OpenMP uses a hermetic libomp prefix produced by
# tools/dependencies/build_openmp.py (installed under .deps/openmp-<ver>-<platform>-<arch>).
# By design we do NOT probe Homebrew/MacPorts/system locations. This helper locates such a
# prefix so USE_OPENMP can default ON out of the box once the runtime has been built.
function(mxnet_apple_discover_openmp_root out_var)
  set(${out_var} "" PARENT_SCOPE)
  file(GLOB _mxnet_openmp_candidates LIST_DIRECTORIES true
       "${CMAKE_CURRENT_SOURCE_DIR}/.deps/openmp-*")
  string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" _mxnet_host_arch)
  set(_arch_match "")
  set(_any_match "")
  foreach(_cand IN LISTS _mxnet_openmp_candidates)
    if(IS_DIRECTORY "${_cand}" AND EXISTS "${_cand}/include/omp.h" AND IS_DIRECTORY "${_cand}/lib")
      if(NOT _any_match)
        set(_any_match "${_cand}")
      endif()
      if(_mxnet_host_arch AND _cand MATCHES "${_mxnet_host_arch}" AND NOT _arch_match)
        set(_arch_match "${_cand}")
      endif()
    endif()
  endforeach()
  if(_arch_match)
    set(${out_var} "${_arch_match}" PARENT_SCOPE)
  elseif(_any_match)
    set(${out_var} "${_any_match}" PARENT_SCOPE)
  endif()
endfunction()

#Some things have order. This must be put in front alone
option(MXNET_BUILD_SHARED_LIBS "Build shared libraries instead of static libraries" ON)
if(APPLE)
  set(MXNET_DEFAULT_USE_CUDA OFF)
  set(MXNET_DEFAULT_USE_OPENCV OFF)
  set(USE_BLAS "apple" CACHE STRING "BLAS Vendor")
  # Default USE_OPENMP ON only when a hermetic libomp prefix is available (a user-supplied
  # OPENMP_ROOT or an auto-discovered .deps/openmp-* tree), so a fresh checkout without the
  # runtime still configures cleanly as a single-threaded build.
  if(OPENMP_ROOT)
    set(MXNET_APPLE_OPENMP_ROOT "${OPENMP_ROOT}")
  else()
    mxnet_apple_discover_openmp_root(MXNET_APPLE_OPENMP_ROOT)
  endif()
  if(MXNET_APPLE_OPENMP_ROOT)
    set(MXNET_DEFAULT_USE_OPENMP ON)
  else()
    set(MXNET_DEFAULT_USE_OPENMP OFF)
  endif()
else()
  set(MXNET_DEFAULT_USE_CUDA ON)
  set(MXNET_DEFAULT_USE_OPENCV ON)
  set(MXNET_DEFAULT_USE_OPENMP ON)
endif()

set(MXNET_DEFAULT_USE_ONEDNN OFF)
if(NOT MSVC AND NOT CMAKE_CROSSCOMPILING)
  if((NOT APPLE) AND (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    set(MXNET_DEFAULT_USE_ONEDNN ON)
  elseif(APPLE)
    set(MXNET_DARWIN_ONEDNN_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
    if(NOT MXNET_DARWIN_ONEDNN_PROCESSOR)
      set(MXNET_DARWIN_ONEDNN_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    if(NOT MXNET_DARWIN_ONEDNN_PROCESSOR)
      set(MXNET_DARWIN_ONEDNN_PROCESSOR "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    string(TOLOWER "${MXNET_DARWIN_ONEDNN_PROCESSOR}" MXNET_DARWIN_ONEDNN_PROCESSOR)
    if(MXNET_DARWIN_ONEDNN_PROCESSOR MATCHES "^(arm64|aarch64)$")
      set(MXNET_DEFAULT_USE_ONEDNN ON)
    endif()
  endif()
endif()
option(USE_CUDA "Build with CUDA support" ${MXNET_DEFAULT_USE_CUDA})
set(MXNET_CUDA_ARCH "8.0;8.6;8.9;9.0;12.0+PTX" CACHE STRING "Target NVIDIA GPU achitecture.
Format: Auto | Common | All | LIST(ARCH_AND_PTX ...)
- \"Auto\" detects local machine GPU compute arch at runtime.
- \"Common\" and \"All\" cover common and entire subsets of architectures
- ARCH_AND_PTX : NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
- NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal Volta Turing
- NUM: Any number. Only those pairs are currently accepted by NVCC though:
       2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2 7.0 7.2 7.5 8.0 8.6 8.9 9.0 12.0
Default targets Ampere (sm_80/sm_86), Ada (sm_89), Hopper (sm_90), and
Blackwell consumer (sm_120) with a forward-compat PTX 12.0 fallback for
future GPUs newer than sm_120.")
option(USE_NCCL "Use NVidia NCCL with CUDA" OFF)
option(USE_OPENCV "Build with OpenCV support" ${MXNET_DEFAULT_USE_OPENCV})
option(USE_OPENMP "Build with Openmp support" ${MXNET_DEFAULT_USE_OPENMP})
option(USE_FATBIN_COMPRESSION "Compress nvcc fatbin output" ON)
cmake_dependent_option(USE_NVML "Build with nvml support if found" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_CUDNN "Build with cudnn support" ON "USE_CUDA" OFF) # one could set CUDNN_ROOT for search path
cmake_dependent_option(USE_CUTENSOR "Build with cuTENSOR support" ON "USE_CUDA" OFF) # one could set CUTENSOR_ROOT for search path
cmake_dependent_option(USE_NVTX "Build with nvtx support if found" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_SSE "Build with x86 SSE instruction support" ON
  "CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL amd64" OFF)
cmake_dependent_option(USE_F16C "Build with x86 F16C instruction support" ON
  "CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL amd64" OFF) # autodetects support if ON
option(USE_LAPACK "Build with lapack support" ON)
option(USE_MKL_LAYERNORM "Use layer normalization from MKL, which is currently slower than internal. No effect unless USE_BLAS=MKL (or mkl)." OFF)
option(USE_ONEDNN "Build with oneDNN support" ${MXNET_DEFAULT_USE_ONEDNN})
cmake_dependent_option(USE_INTGEMM "Build with x86_64 intgemm library for low-precision multiplication" ON "CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64" OFF)
if(NOT MSVC)
  option(USE_OPERATOR_TUNING  "Enable auto-tuning of operators" ON)
else()
  option(USE_OPERATOR_TUNING  "Enable auto-tuning of operators" OFF)
endif()
option(USE_GPERFTOOLS "Build with GPerfTools support" OFF)
option(USE_JEMALLOC "Build with Jemalloc support" OFF)
option(USE_LIBJPEG_TURBO "Use libjpeg-turbo" OFF)
option(USE_DIST_KVSTORE "Build with DIST_KVSTORE support" OFF)
option(USE_PLUGINS_WARPCTC "Use WARPCTC Plugins" OFF)
option(USE_CPP_PACKAGE "Build C++ Package" OFF)
option(USE_MXNET_LIB_NAMING "Use MXNet library naming conventions." ON)
option(USE_GPROF "Compile with gprof (profiling) flag" OFF)
option(USE_VTUNE "Enable use of Intel Amplifier XE (VTune)" OFF) # one could set VTUNE_ROOT for search path
option(USE_TVM_OP "Enable use of TVM operator build system." OFF)
option(BUILD_CPP_EXAMPLES "Build cpp examples" ON)
option(INSTALL_EXAMPLES "Install the example source files." OFF)
option(USE_SIGNAL_HANDLER "Print stack traces on segfaults." ON)
option(USE_TENSORRT "Enable inference optimization with TensorRT." OFF)
option(USE_ASAN "Enable Clang/GCC ASAN sanitizers." OFF)
cmake_dependent_option(ENABLE_TESTCOVERAGE "Enable compilation with test coverage metric output" OFF "NOT MSVC" OFF)
option(BUILD_EXTENSION_PATH "Path to extension to build" "")
option(BUILD_CYTHON_MODULES "Build cython modules." OFF)
option(COLORIZE_OUTPUT "Colorize output during compilation" ON)
option(LOG_FATAL_THROW "Log exceptions but do not abort" ON)
cmake_dependent_option(USE_SPLIT_ARCH_DLL "Build a separate DLL for each Cuda arch (Windows only)." ON "MSVC" OFF)
cmake_dependent_option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON "UNIX" OFF)
cmake_dependent_option(MXNET_FORCE_SHARED_CRT "Build with dynamic CRT on Windows (/MD)" ON "MXNET_BUILD_SHARED_LIBS" OFF)

if(DEFINED USE_AZURE AND USE_AZURE)
  message(FATAL_ERROR
    "USE_AZURE is not supported by this MXNet CMake build: the vendored "
    "dmlc-core Azure filesystem is incomplete and Azure SDK dependencies are "
    "not wired. Disable USE_AZURE or add full Azure SDK integration before "
    "enabling it.")
endif()

message(STATUS "CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING}")
message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")

message(STATUS "CMAKE_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}")

find_package(Git QUIET)
if(${GIT_FOUND})
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD 
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    RESULT_VARIABLE BRANCH_FAILED
  )
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT
    RESULT_VARIABLE COMMIT_FAILED
  )
  if(NOT BRANCH_FAILED)
    string(REGEX REPLACE "\n$" "" GIT_BRANCH "${GIT_BRANCH}")
    add_compile_definitions(MXNET_BRANCH="${GIT_BRANCH}")
  else()
    add_compile_definitions(MXNET_BRANCH="Unavailable")
  endif()
  if(NOT COMMIT_FAILED)
    string(REGEX REPLACE "\n$" "" GIT_COMMIT "${GIT_COMMIT}")
    add_compile_definitions(MXNET_COMMIT_HASH="${GIT_COMMIT}")
  else()
    add_compile_definitions(MXNET_COMMIT_HASH="Unavailable")
  endif()
endif()

if(USE_TVM_OP)
  add_definitions(-DMXNET_USE_TVM_OP=1)
endif()

if(MXNET_FORCE_SHARED_CRT)
  set(DMLC_FORCE_SHARED_CRT ON)
  set(gtest_force_shared_crt ON)
endif()

message(STATUS "CMake version '${CMAKE_VERSION}' using generator '${CMAKE_GENERATOR}'")
if(USE_CUDA)
  cmake_minimum_required(VERSION 3.13.2)  # CUDA 10 (Turing) detection available starting 3.13.2
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING
        "MXNet sets CUDA -gencode flags from MXNET_CUDA_ARCH")
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER AND UNIX AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(ENV{CUDACXX} "/usr/local/cuda/bin/nvcc")
    message(WARNING "CMAKE_CUDA_COMPILER guessed: " $ENV{CUDACXX} "\n"
      "Please fix your cuda installation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#mandatory-post")
  endif()
  enable_language(CUDA)
  # CUDA 13's CCCL (libcu++, Thrust, CUB) hard-errors on anything below C++17.
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(UNIX)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  else()
    message(STATUS "Could not find CCache. Consider installing CCache to speed up compilation.")
  endif()
endif()

if(MSVC)
  set(SYSTEM_ARCHITECTURE x86_64)
  enable_language(ASM_MASM)
else()
  execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE SYSTEM_ARCHITECTURE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Distribution")
  if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH $\{ORIGIN\})
    # Enforce DT_PATH instead of DT_RUNPATH
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")
  endif()
  set(Protobuf_USE_STATIC_LIBS ON)
endif()

set(MXNET_UPSTREAM_CMAKE_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/upstream")
set(MXNET_CMAKE_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
set(CMAKE_MODULE_PATH "${MXNET_UPSTREAM_CMAKE_MODULE_DIR};${MXNET_CMAKE_MODULE_DIR};${CMAKE_MODULE_PATH}")

SET(EXTRA_OPERATORS "" CACHE PATH "EXTRA OPERATORS PATH")

if("$ENV{VERBOSE}" STREQUAL "1")
  message(STATUS " Verbose Makefile ACTIVATED")
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

#Switch off modern thread local for dmlc-core, please see: https://github.com/dmlc/dmlc-core/issues/571#issuecomment-543467484
add_definitions(-DDMLC_MODERN_THREAD_LOCAL=0)
# disable stack trace in exception by default.
add_definitions(-DDMLC_LOG_STACK_TRACE_SIZE=0)

add_definitions(-DDMLC_USE_CXX11)
add_definitions(-DDMLC_STRICT_CXX11)
add_definitions(-DDMLC_USE_CXX14)
add_definitions(-DMSHADOW_IN_CXX11)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if(${COLORIZE_OUTPUT})
    string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")
  endif()
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
  if(${COLORIZE_OUTPUT})
    string(APPEND CMAKE_CXX_FLAGS " -fdiagnostics-color=always")
  endif()
endif()

if(MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DMXNET_EXPORTS)
  add_definitions(-DNNVM_EXPORTS)
  add_definitions(-DNOMINMAX)
  set(CMAKE_C_FLAGS "/MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS} /bigobj")
else()
  include(CheckCXXCompilerFlag)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-sign-compare")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_ASSERTIONS")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_ASSERTIONS")
  else()
    add_definitions(-DNDEBUG=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}")
endif()

if(NOT mxnet_LINKER_LIBS)
  set(mxnet_LINKER_LIBS "")
endif()

if(USE_GPROF)
  message(STATUS "Using GPROF")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_LINK_LIBRARY_FILE_FLAG "${CMAKE_LINK_LIBRARY_FILE_FLAG} -g -pg")
endif()

if(USE_VTUNE)
  message(STATUS "Using VTUNE")
  if(NOT VTUNE_ROOT)
    set(VTUNE_ROOT /opt/intel/vtune_amplifier_xe_2017)
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_LINK_LIBRARY_FILE_FLAG "${CMAKE_LINK_LIBRARY_FILE_FLAG} -g -pg")
  add_definitions(-DMXNET_USE_VTUNE=1)
  include_directories(${VTUNE_ROOT}/include)
  list(APPEND mxnet_LINKER_LIBS ${VTUNE_ROOT}/lib64/libittnotify.a)
  list(APPEND mxnet_LINKER_LIBS dl)
endif()

if(USE_TENSORRT)
  message(STATUS "Using TensorRT")
  set(ONNX_PATH 3rdparty/onnx-tensorrt/third_party/onnx/build/)
  set(ONNX_TRT_PATH 3rdparty/onnx-tensorrt/build/)

  include_directories(${ONNX_PATH})
  include_directories(3rdparty/onnx-tensorrt/)
  include_directories(3rdparty/)
  include_directories(3rdparty/onnx-tensorrt/third_party/onnx/)
  add_definitions(-DMXNET_USE_TENSORRT=1)
  add_definitions(-DONNX_NAMESPACE=onnx)
  add_definitions(-DONNX_ML=1)

  find_package(Protobuf REQUIRED)

  find_library(ONNX_LIBRARY NAMES libonnx.so REQUIRED
          PATHS ${ONNX_PATH}
          DOC "Path to onnx library.")
  find_library(ONNX_PROTO_LIBRARY NAMES libonnx_proto.so REQUIRED
          PATHS ${ONNX_PATH}
          DOC "Path to onnx_proto library.")
  find_library(ONNX_TRT_PARSER_LIBRARY NAMES libnvonnxparser.so REQUIRED
          PATHS ${ONNX_TRT_PATH}
          DOC "Path to onnx_proto parser library.")

  list(APPEND mxnet_LINKER_LIBS libnvinfer.so ${ONNX_TRT_PARSER_LIBRARY}
          ${ONNX_PROTO_LIBRARY} ${ONNX_LIBRARY} ${PROTOBUF_LIBRARY})
endif()

include(GNUInstallDirs)
if(USE_MKL_LAYERNORM)
  add_definitions(-DMXNET_USE_MKL_LAYERNORM=1)
endif()

set(OPENMP_ROOT "${MXNET_APPLE_OPENMP_ROOT}" CACHE PATH "OpenMP runtime install prefix")
function(mxnet_prepare_openmp_root)
  if(NOT OPENMP_ROOT)
    return()
  endif()

  unset(MXNET_OPENMP_INCLUDE_DIR CACHE)
  unset(MXNET_OPENMP_LIBRARY CACHE)
  unset(OpenMP_C_INCLUDE_DIR CACHE)
  unset(OpenMP_CXX_INCLUDE_DIR CACHE)
  unset(OpenMP_omp_LIBRARY CACHE)
  find_path(MXNET_OPENMP_INCLUDE_DIR
            NAMES omp.h
            PATHS "${OPENMP_ROOT}/include"
            NO_DEFAULT_PATH)
  find_library(MXNET_OPENMP_LIBRARY
               NAMES omp libomp
               PATHS "${OPENMP_ROOT}/lib" "${OPENMP_ROOT}/lib64"
               NO_DEFAULT_PATH)

  if(MXNET_OPENMP_INCLUDE_DIR)
    include_directories(SYSTEM "${MXNET_OPENMP_INCLUDE_DIR}")
    set(OpenMP_C_INCLUDE_DIR "${MXNET_OPENMP_INCLUDE_DIR}" CACHE PATH "OpenMP C include directory" FORCE)
    set(OpenMP_CXX_INCLUDE_DIR "${MXNET_OPENMP_INCLUDE_DIR}" CACHE PATH "OpenMP CXX include directory" FORCE)
  endif()
  if(MXNET_OPENMP_LIBRARY)
    set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "OpenMP C library names" FORCE)
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP CXX library names" FORCE)
    set(OpenMP_omp_LIBRARY "${MXNET_OPENMP_LIBRARY}" CACHE FILEPATH "OpenMP runtime library" FORCE)
  endif()
  if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # AppleClang needs "-Xpreprocessor -fopenmp" to honor #pragma omp. Also pass an explicit
    # -I: CMake's FindOpenMP verification compiles an isolated try_compile that does not see
    # include_directories(), so omp.h must be reachable through the flags themselves.
    set(_mxnet_apple_omp_flags "-Xpreprocessor -fopenmp")
    if(MXNET_OPENMP_INCLUDE_DIR)
      set(_mxnet_apple_omp_flags "${_mxnet_apple_omp_flags} -I${MXNET_OPENMP_INCLUDE_DIR}")
    endif()
    set(OpenMP_C_FLAGS "${_mxnet_apple_omp_flags}" CACHE STRING "OpenMP C compiler flags" FORCE)
    set(OpenMP_CXX_FLAGS "${_mxnet_apple_omp_flags}" CACHE STRING "OpenMP CXX compiler flags" FORCE)
  endif()
endfunction()
if(USE_OPENMP)
  mxnet_prepare_openmp_root()
endif()

if(USE_ONEDNN)
  # CPU architecture (e.g., C5) can't run on another architecture (e.g., g3).
  if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /Gy")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /EHsc /Gy")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /EHsc /Gy")
    if(NOT MXNET_FORCE_SHARED_CRT)
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
      set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
    endif()
  endif()

  function(load_onednn)
    set(ONEDNN_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
    set(ONEDNN_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
    set(ONEDNN_ARCH_OPT_FLAGS "" CACHE INTERNAL "" FORCE)
    set(ONEDNN_ENABLE_JIT_PROFILING OFF CACHE INTERNAL "" FORCE)
    set(ONEDNN_LIBRARY_TYPE STATIC CACHE INTERNAL "" FORCE)
    set(ONEDNN_ENABLE_CONCURRENT_EXEC ON CACHE INTERNAL "" FORCE)
    set(ONEDNN_ENABLE_PRIMITIVE_CACHE ON CACHE INTERNAL "" FORCE)

    if(NOT USE_OPENMP)
      set(ONEDNN_CPU_RUNTIME SEQ CACHE INTERNAL "" FORCE)
    endif()

    set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/onednn")
    add_subdirectory(3rdparty/onednn)
  endfunction()
  load_onednn()
  include_directories(3rdparty/onednn/include)
  include_directories(${PROJECT_BINARY_DIR}/3rdparty/onednn/include)
  add_definitions(-DMXNET_USE_ONEDNN=1)
  list(APPEND mxnet_LINKER_LIBS dnnl)
  set_target_properties(dnnl PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
endif()

if(USE_CPP_PACKAGE)
    add_definitions(-DMXNET_USE_CPP_PACKAGE=1)
endif()

if(USE_INTGEMM)
  message(STATUS "Using intgemm")
  add_subdirectory(3rdparty/intgemm EXCLUDE_FROM_ALL)
  add_definitions(-DMXNET_USE_INTGEMM=1)
endif()

# Allow Cuda compiles outside of src tree to find things in 'src' and 'include'
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

cmake_dependent_option(USE_INT64_TENSOR_SIZE "Use int64_t to represent the total number of elements in a tensor" ON "CMAKE_SIZEOF_VOID_P EQUAL 8" OFF)

include(cmake/ChooseBlas.cmake)

if(USE_ASAN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
  set(GTEST_LIBRARIES "${GTEST_LIBRARIES} -fsanitize=address")
  list(APPEND mxnet_LINKER_LIBS asan)
endif()

list(APPEND mxnet_LINKER_LIBS ${mshadow_LINKER_LIBS})
message("After choosing blas, linking to ${mxnet_LINKER_LIBS}")

foreach(var ${C_CXX_INCLUDE_DIRECTORIES})
    include_directories(${var})
endforeach()

include_directories("include")
include_directories("3rdparty/tvm/nnvm/include")
include_directories("3rdparty/tvm/include")
include_directories("3rdparty/dmlc-core/include")
include_directories("3rdparty/dlpack/include")

if(UNIX)
  find_library(RTLIB rt)
  if(RTLIB)
    list(APPEND mxnet_LINKER_LIBS ${RTLIB})
  endif()
endif()

set(ALT_MALLOC_FLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

# ---[ gperftools
if(USE_GPERFTOOLS)
  find_package(Gperftools)
  if(GPERFTOOLS_FOUND)
    message(STATUS "Using Gperftools malloc (tcmalloc)")
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${GPERFTOOLS_LIBRARIES})
    set(USE_JEMALLOC 0)
  endif()
endif()

# ---[ jemalloc
if(USE_JEMALLOC)
  if(GPERFTOOLS_FOUND)
    message(ERROR " Only one of USE_JEMALLOC and USE_GPERFTOOLS can be defined at once")
  endif()
  find_package(JeMalloc)
  if(JEMALLOC_FOUND)
    message(STATUS "Using JEMalloc malloc")
    add_definitions(-DUSE_JEMALLOC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALT_MALLOC_FLAGS}")
    include_directories(${JEMALLOC_INCLUDE_DIRS})
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${JEMALLOC_LIBRARIES})
  endif()
endif()

if(USE_LIBJPEG_TURBO)
  set(TURBOJPEG_ROOT "" CACHE PATH "libjpeg-turbo install prefix")
  if(TURBOJPEG_ROOT)
    find_path(TURBOJPEG_INCLUDE_DIR
              NAMES turbojpeg.h
              PATHS "${TURBOJPEG_ROOT}/include"
              NO_DEFAULT_PATH)
    find_library(TURBOJPEG_LIBRARY
                 NAMES turbojpeg libturbojpeg
                 PATHS "${TURBOJPEG_ROOT}/lib" "${TURBOJPEG_ROOT}/lib64"
                 NO_DEFAULT_PATH)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(TurboJPEG
                                      REQUIRED_VARS TURBOJPEG_INCLUDE_DIR TURBOJPEG_LIBRARY)
    include_directories(SYSTEM ${TURBOJPEG_INCLUDE_DIR})
    list(APPEND mxnet_LINKER_LIBS ${TURBOJPEG_LIBRARY})
    message(STATUS "libjpeg-turbo found (${TURBOJPEG_LIBRARY})")
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_search_module(TURBOJPEG libturbojpeg)
    endif()
    if(TURBOJPEG_FOUND)
      include_directories(SYSTEM ${TURBOJPEG_INCLUDE_DIRS})
      list(APPEND mxnet_LINKER_LIBS ${TURBOJPEG_LINK_LIBRARIES})
      message(STATUS "libjpeg-turbo found (${TURBOJPEG_LINK_LIBRARIES})")
    else()
      find_path(TURBOJPEG_INCLUDE_DIR NAMES turbojpeg.h)
      find_library(TURBOJPEG_LIBRARY NAMES turbojpeg libturbojpeg)
      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(TurboJPEG
                                        REQUIRED_VARS TURBOJPEG_INCLUDE_DIR TURBOJPEG_LIBRARY)
      include_directories(SYSTEM ${TURBOJPEG_INCLUDE_DIR})
      list(APPEND mxnet_LINKER_LIBS ${TURBOJPEG_LIBRARY})
      message(STATUS "libjpeg-turbo found (${TURBOJPEG_LIBRARY})")
    endif()
  endif()
  add_definitions(-DMXNET_USE_LIBJPEG_TURBO=1)
else()
  add_definitions(-DMXNET_USE_LIBJPEG_TURBO=0)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  set(OPENCV_ROOT "" CACHE PATH "OpenCV install path. Supports autodetection.")
  if(OPENCV_ROOT)
    find_package(OpenCV
                 CONFIG
                 COMPONENTS core imgproc imgcodecs
                 PATHS "${OPENCV_ROOT}" "${OPENCV_ROOT}/lib/cmake/opencv4"
                 NO_DEFAULT_PATH)
  else()
    find_package(OpenCV COMPONENTS core imgproc imgcodecs)
  endif()
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    message(STATUS "OpenCV imgcodecs missing; trying OpenCV 2.x highgui codec layout")
    if(OPENCV_ROOT)
      find_package(OpenCV
                   REQUIRED
                   CONFIG
                   COMPONENTS core highgui imgproc
                   PATHS "${OPENCV_ROOT}" "${OPENCV_ROOT}/lib/cmake/opencv4"
                   NO_DEFAULT_PATH)
    else()
      find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
    endif()
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  set(MXNET_OPENCV_LINK_LIBS ${OpenCV_LIBS})
  if(OpenCV_VERSION_MAJOR GREATER 2 AND TARGET opencv_imgcodecs)
    get_target_property(MXNET_OPENCV_IMGCODECS_DEPS opencv_imgcodecs INTERFACE_LINK_LIBRARIES)
    if(MXNET_OPENCV_IMGCODECS_DEPS MATCHES "(^|;)opencv_core(;|$)" AND
       MXNET_OPENCV_IMGCODECS_DEPS MATCHES "(^|;)opencv_imgproc(;|$)")
      set(MXNET_OPENCV_LINK_LIBS opencv_imgcodecs)
    endif()
  endif()
  list(APPEND mxnet_LINKER_LIBS ${MXNET_OPENCV_LINK_LIBS})
  message(STATUS "OpenCV ${OpenCV_VERSION} found (${OpenCV_CONFIG_PATH})")
  message(STATUS " OpenCV_LIBS=${OpenCV_LIBS}")
  message(STATUS " MXNet OpenCV link libs=${MXNET_OPENCV_LINK_LIBS}")
  add_definitions(-DMXNET_USE_OPENCV=1)
else(USE_OPENCV)
  message(STATUS "OpenCV Disabled")
  add_definitions(-DMXNET_USE_OPENCV=0)
endif()

# ---[ OpenMP
if(USE_OPENMP)
  if(APPLE AND NOT OpenMP_omp_LIBRARY AND NOT OPENMP_ROOT)
    message(FATAL_ERROR
      "USE_OPENMP=ON on macOS but no libomp runtime was found.\n"
      "Build the repo-local runtime once:\n"
      "    python tools/dependencies/build_openmp.py\n"
      "It installs under .deps/ and is auto-discovered on reconfigure. Alternatively pass\n"
      "-DOPENMP_ROOT=<prefix>, or set -DUSE_OPENMP=OFF to build single-threaded.")
  endif()
  find_package(OpenMP REQUIRED COMPONENTS C CXX)
  if(OpenMP_FOUND)
    include_directories(SYSTEM ${OpenMP_CXX_INCLUDE_DIRS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # Enable pragma omp simd
    # "While the name of this switch is 'experimental', the switch itself, and
    # the functionality it enables is fully supported and production-ready.
    # The name reflects that it doesn’t enable any complete subset or
    # version of an OpenMP standard."
    # -- https://devblogs.microsoft.com/cppblog/simd-extension-to-c-openmp-in-visual-studio/
    if(MSVC)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -openmp:experimental")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:experimental")
    endif()
    # NOTE: CMake's FindOpenMP defines OpenMP_<lang>_FLAGS and the imported target
    # OpenMP::OpenMP_CXX, but never an OpenMP_EXE_LINKER_FLAGS variable, so the old
    # set() calls here expanded to nothing. Linking is handled by appending
    # OpenMP::OpenMP_CXX to mxnet_LINKER_LIBS below.
    if(TARGET OpenMP::OpenMP_CXX)
      list(APPEND mxnet_LINKER_LIBS OpenMP::OpenMP_CXX)
    else()
      list(APPEND mxnet_LINKER_LIBS ${OpenMP_CXX_LIBRARIES})
    endif()
    add_definitions(-DMXNET_USE_OPENMP=1)
  endif()
elseif(UNIX)
  # Prefer compiler pthread flag. This is the recommended way, but not backwards
  # compatible and thus not the cmake default.
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  list(APPEND mxnet_LINKER_LIBS Threads::Threads)
endif()

# ---[ LAPack
if(USE_LAPACK)
  message("USE_LAPACK is ON")
  add_definitions(-DMXNET_USE_LAPACK=1)
  if(NOT USE_LAPACKE_INTERFACE)
    # BLAS=open case is handled in ChooseBlas.cmake
    if(NOT MSVC AND NOT CMAKE_BUILD_TYPE STREQUAL "Distribution"
       AND NOT BLAS STREQUAL "Open" AND NOT BLAS STREQUAL "open"
       AND NOT BLAS STREQUAL "apple")
      list(APPEND mxnet_LINKER_LIBS lapack)
    endif()
  endif()
endif()

# ---[ jemalloc
if(USE_JEMALLOC)
  find_package(JeMalloc)
  if(JEMALLOC_FOUND)
    add_definitions(-DUSE_JEMALLOC)
    include_directories(${JEMALLOC_INCLUDE_DIRS})
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${JEMALLOC_LIBRARIES})
  endif()
endif()

include(CTest)
set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest")
set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/googletest/include")
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_LIBRARY gtest)
set(BUILD_GMOCK OFF)

add_subdirectory(${GTEST_ROOT})
set_target_properties(gtest PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
set_target_properties(gtest_main PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
find_package(GTest REQUIRED)

# cudnn detection
if(USE_CUDNN)
  find_package(CUDNN)
  if(CUDNN_FOUND)
    add_definitions(-DUSE_CUDNN)
    include_directories(SYSTEM ${CUDNN_INCLUDE})
    list(APPEND mxnet_LINKER_LIBS ${CUDNN_LIBRARY})
  else()
    set(USE_CUDNN OFF)
  endif()
endif()

# cutensor detection
if(USE_CUTENSOR)
  find_package(CUTENSOR)
  if(CUTENSOR_FOUND)
    add_definitions(-DUSE_CUTENSOR)
    include_directories(SYSTEM ${CUTENSOR_INCLUDE})
    list(APPEND mxnet_LINKER_LIBS ${CUTENSOR_LIBRARY})
  else()
    set(USE_CUTENSOR OFF)
  endif()
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/cmake)
  add_subdirectory("3rdparty/dmlc-core")
  set_target_properties(dmlc PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
endif()

# CONFIGURE_DEPENDS: re-glob at build time so adding/removing a source file
# triggers a reconfigure instead of silently producing a stale incremental build.
FILE(GLOB_RECURSE SOURCE CONFIGURE_DEPENDS "src/*.cc" "src/*.h" "include/*.h")
FILE(GLOB_RECURSE CUDA CONFIGURE_DEPENDS "src/*.cu" "src/*.cuh")

if(NOT (USE_OPERATOR_TUNING AND USE_OPENMP))
  list(REMOVE_ITEM SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/operator/operator_tune.cc")
endif()

if(MSVC)
  FILE(GLOB_RECURSE TVM_BRIDGE_SOURCE "src/*/tvm_bridge.cc")
  list(REMOVE_ITEM SOURCE ${TVM_BRIDGE_SOURCE})
endif()

if(NOT USE_INTGEMM)
  FILE(GLOB_RECURSE INTGEMM_OPERATOR_SOURCE "src/operator/contrib/intgemm/*.cc" "src/operator/contrib/intgemm/*.h")
  list(REMOVE_ITEM SOURCE ${INTGEMM_OPERATOR_SOURCE})
endif()

# add nnvm to source
FILE(GLOB_RECURSE NNVMSOURCE
  3rdparty/tvm/nnvm/src/c_api/*.cc
  3rdparty/tvm/nnvm/src/core/*.cc
  3rdparty/tvm/nnvm/src/pass/*.cc
  3rdparty/tvm/nnvm/src/c_api/*.h
  3rdparty/tvm/nnvm/src/core/*.h
  3rdparty/tvm/nnvm/src/pass/*.h
  3rdparty/tvm/nnvm/include/*.h)
add_library(nnvm OBJECT ${NNVMSOURCE})
set_target_properties(nnvm PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(nnvm PRIVATE -Wno-deprecated-declarations -Wno-unused-value)
endif()
list(APPEND SOURCE $<TARGET_OBJECTS:nnvm>)

add_library(miniz STATIC "3rdparty/miniz/miniz.c")
target_include_directories(miniz PUBLIC "3rdparty/miniz")
list(APPEND mxnet_LINKER_LIBS miniz)

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc" "3rdparty/tvm/nnvm/*.cc" "plugin/*.cc")
FILE(GLOB_RECURSE GROUP_Include "src/*.h" "3rdparty/tvm/nnvm/*.h" "3rdparty/mshadow/mshadow/*.h" "plugin/*.h")
FILE(GLOB_RECURSE GROUP_CUDA "src/*.cu" "src/*.cuh" "3rdparty/mshadow/mshadow/*.cuh" "plugin/*.cu"
  "plugin/*.cuh" "3rdparty/nvidia_cub/cub/*.cuh")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_Include})
assign_source_group("CUDA" ${GROUP_CUDA})

if(USE_PLUGINS_WARPCTC)
    set(WARPCTC_INCLUDE "" CACHE PATH "WARPCTC include directory containing ctc.h")
    set(WARPCTC_LIB "" CACHE FILEPATH "WARPCTC library")
    set(WARPCTC_LIB_DEBUG "" CACHE FILEPATH "WARPCTC debug library")
    set(WARPCTC_LIB_RELEASE "" CACHE FILEPATH "WARPCTC release library")
    if(NOT WARPCTC_INCLUDE)
      find_path(WARPCTC_INCLUDE ctc.h)
    endif()
    if(NOT WARPCTC_LIB)
      if(CMAKE_BUILD_TYPE MATCHES "[Dd]ebug" AND WARPCTC_LIB_DEBUG)
        set(WARPCTC_LIB "${WARPCTC_LIB_DEBUG}")
      elseif(WARPCTC_LIB_RELEASE)
        set(WARPCTC_LIB "${WARPCTC_LIB_RELEASE}")
      elseif(WARPCTC_LIB_DEBUG)
        set(WARPCTC_LIB "${WARPCTC_LIB_DEBUG}")
      else()
        find_library(WARPCTC_LIB NAMES warpctc)
      endif()
    endif()
    if(NOT WARPCTC_INCLUDE OR NOT WARPCTC_LIB)
      message(FATAL_ERROR
        "USE_PLUGINS_WARPCTC=ON requires WARPCTC_INCLUDE and WARPCTC_LIB "
        "(or discoverable ctc.h/libwarpctc).")
    endif()
    include_directories(SYSTEM ${WARPCTC_INCLUDE})
    list(APPEND mxnet_LINKER_LIBS ${WARPCTC_LIB})
    FILE(GLOB_RECURSE PLUGINS_SOURCE "plugin/warpctc/*.cc" "plugin/warpctc/*.h")
    FILE(GLOB_RECURSE PLUGINS_CUSRC "plugin/warpctc/*.cu")
    list(APPEND SOURCE ${PLUGINS_SOURCE})
    list(APPEND CUDA ${PLUGINS_CUSRC})
endif()

if(USE_OPERATOR_TUNING AND USE_OPENMP)
  add_definitions(-DMXNET_USE_OPERATOR_TUNING=1)
endif()

if(NOT (EXTRA_OPERATORS STREQUAL ""))
    mxnet_source_group("Extra"   GLOB_RECURSE "${EXTRA_OPERATORS}/*.cc")
    mxnet_source_group("Extra\\Cuda"   GLOB_RECURSE "${EXTRA_OPERATORS}/*.cu")
    FILE(GLOB_RECURSE EXTRA_SRC "${EXTRA_OPERATORS}/*.cc")
    FILE(GLOB_RECURSE EXTRA_CUSRC "${EXTRA_OPERATORS}/*.cu")
    list(APPEND SOURCE ${EXTRA_SRC} ${EXTRA_CUSRC})
endif()

if(MSVC AND NOT MXNET_FORCE_SHARED_CRT)
  foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    elseif(${flag_var} MATCHES "/MDd")
      string(REGEX REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")    
    endif()
  endforeach(flag_var)
endif()

if(USE_CUDA)
  # CUDA_SELECT_NVCC_ARCH_FLAGS is not deprecated, though part of deprecated
  # FindCUDA https://gitlab.kitware.com/cmake/cmake/issues/19199
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/upstream/select_compute_arch.cmake)
  CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${MXNET_CUDA_ARCH})
  message("-- CUDA: Using the following NVCC architecture flags ${CUDA_ARCH_FLAGS}")
  set(arch_code_list)
  foreach(arch_str ${CUDA_ARCH_FLAGS})
    if((arch_str MATCHES ".*sm_[0-9]+"))
      string( REGEX REPLACE  ".*sm_([0-9]+)" "\\1" arch_code ${arch_str} )
      list(APPEND arch_code_list ${arch_code})
    endif()
  endforeach()

  # H4: fail loudly if the toolkit silently dropped an explicitly requested numeric
  # arch (e.g. sm_90/sm_120 on a CUDA older than 11.8/12.8). CUDA_SELECT_NVCC_ARCH_FLAGS
  # filters unknown arches out, so without this check the wheel would ship with no
  # kernel image / no PTX fallback and fail at runtime ("no kernel image is available").
  set(arch_ptx_list)
  foreach(arch_str ${CUDA_ARCH_FLAGS})
    if(arch_str MATCHES ".*compute_[0-9]+")
      string(REGEX REPLACE ".*compute_([0-9]+)" "\\1" ptx_code ${arch_str})
      list(APPEND arch_ptx_list ${ptx_code})
    endif()
  endforeach()
  foreach(req_arch ${MXNET_CUDA_ARCH})
    # Only validate explicit numeric requests like "8.9" or "12.0+PTX" -- skip the
    # toolkit-defined sets (Auto/Common/All) and NAME forms (Ampere/Hopper/...).
    if(req_arch MATCHES "^[0-9]+\\.[0-9]+")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\1\\2" req_code "${req_arch}")
      list(FIND arch_code_list "${req_code}" _sm_idx)
      list(FIND arch_ptx_list "${req_code}" _ptx_idx)
      if(_sm_idx EQUAL -1 AND _ptx_idx EQUAL -1)
        message(FATAL_ERROR
          "Requested CUDA arch '${req_arch}' (sm_${req_code}) produced no -gencode flags. "
          "Your CUDA toolkit (version ${CUDAToolkit_VERSION}) is too old to know this "
          "architecture, so the binary would have no kernel image for it. Upgrade CUDA or "
          "set MXNET_CUDA_ARCH to architectures your toolkit supports.")
      endif()
    endif()
  endforeach()

  string(REPLACE ";" " " CUDA_ARCH_FLAGS_SPACES "${CUDA_ARCH_FLAGS}")

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    set(MXNET_CMAKE_MODULE_PATH_WITH_UPSTREAM "${CMAKE_MODULE_PATH}")
    list(REMOVE_ITEM CMAKE_MODULE_PATH "${MXNET_UPSTREAM_CMAKE_MODULE_DIR}")
  endif()
  find_package(CUDAToolkit REQUIRED cublas cufft cusolver curand nvrtc
    OPTIONAL_COMPONENTS nvToolsExt)
  if(DEFINED MXNET_CMAKE_MODULE_PATH_WITH_UPSTREAM)
    set(CMAKE_MODULE_PATH "${MXNET_CMAKE_MODULE_PATH_WITH_UPSTREAM}")
    unset(MXNET_CMAKE_MODULE_PATH_WITH_UPSTREAM)
  endif()

  # CUDA::cuda_driver links the CUDA driver stub (SONAME libcuda.so.1), which
  # supplies the Driver-API symbols libmxnet references directly (cuLaunchKernel,
  # cuModuleLoad, ...).  Without it those stay UNDEFINED with no DT_NEEDED for
  # libcuda.so.1, so a clean wheel venv aborts dlopen with
  # "undefined symbol: cuLaunchKernel" (the dev tree happens to have the driver
  # loaded globally).  The stub is host-provided at runtime by the NVIDIA driver
  # (never bundled).  Emitting the NEEDED at link time is robust; patchelf
  # --add-needed corrupts this ~1 GB binary.
  list(APPEND mxnet_LINKER_LIBS CUDA::cudart CUDA::cublas CUDA::cufft CUDA::cusolver CUDA::curand
                                CUDA::nvrtc CUDA::cuda_driver)
  list(APPEND SOURCE ${CUDA})
  add_definitions(-DMXNET_USE_CUDA=1)

  if(UNIX)
    if(USE_NVML)
      find_package(NVML)
      if(NVML_FOUND)
        include_directories(${NVML_INCLUDE_DIRS})
        list(APPEND mxnet_LINKER_LIBS ${NVML_LIBRARIES})
        add_definitions(-DMXNET_USE_NVML=1)
      else()
        add_definitions(-DMXNET_USE_NVML=0)
        message(WARNING "Could not find NVML libraries")
      endif()
    endif()
  endif()
  if(USE_NCCL)
    find_package(NCCL)
    if(NCCL_FOUND)
      include_directories(${NCCL_INCLUDE_DIRS})
      list(APPEND mxnet_LINKER_LIBS ${NCCL_LIBRARIES})
      add_definitions(-DMXNET_USE_NCCL=1)
    else()
      message(FATAL_ERROR
        "USE_NCCL=ON but NCCL was not found. Install libnccl-dev or set "
        "NCCL_ROOT/NCCL_INCLUDE_DIR/NCCL_LIB_DIR to the NCCL headers and library.")
    endif()
  endif()
  if(UNIX)
    if(USE_NVTX AND CUDA_nvToolsExt_LIBRARY)
      list(APPEND mxnet_LINKER_LIBS CUDA::nvToolsExt)
      add_definitions(-DMXNET_USE_NVTX=1)
    else()
      message("Building without NVTX support.")
    endif()
  endif()

  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  # CUDA 13 relocated the CCCL umbrella (thrust, cub, libcudacxx) under
  # <toolkit>/include/cccl/. Host-side .cc files that #include <thrust/...>
  # need this path on the include search list.
  foreach(_cuda_inc ${CUDAToolkit_INCLUDE_DIRS})
    if(EXISTS "${_cuda_inc}/cccl/thrust")
      include_directories("${_cuda_inc}/cccl")
    endif()
  endforeach()
  link_directories(${CUDAToolkit_LIBRARY_DIR})
endif()

if(CUDAToolkit_VERSION_MAJOR LESS "11")
  include_directories("3rdparty/nvidia_cub")
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /Gy")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /EHsc /Gy")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /EHsc /Gy")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")
  set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL} /OPT:REF /OPT:ICF")
  set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF /OPT:ICF")

endif()

# Add cmake targets
add_subdirectory("3rdparty/mshadow")

if(USE_CUDA AND USE_FATBIN_COMPRESSION)
  string(APPEND CMAKE_CUDA_FLAGS " --fatbin-options --compress-all")
  message("-- CUDA: Adding NVCC options: --fatbin-options --compress-all")
endif()
if(UNIX)
  string(APPEND CMAKE_CUDA_FLAGS " ${CUDA_ARCH_FLAGS_SPACES}")
  # Create dummy file since we want an empty shared library before linking
  set(DUMMY_SOURCE ${CMAKE_BINARY_DIR}/dummy.c)
  file(WRITE ${DUMMY_SOURCE} "")
  if(MXNET_BUILD_SHARED_LIBS)
    add_library(mxnet SHARED ${SOURCE})
  else()
    add_library(mxnet STATIC ${SOURCE})
  endif()
  target_link_libraries(mxnet PUBLIC mshadow)
  target_link_libraries(mxnet PUBLIC ${CMAKE_DL_LIBS})
  if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(mxnet PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
    # Ignore erroneous compiler warnings:
    # 1) variables used in '#pragma omp parallel' are considered unused
    target_compile_options(mxnet PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-variable>")
    # CN2 policy: GCC 13 emits noisy -Warray-bounds false positives against
    # the deliberate flexible-tail allocation used by InplaceArrayBase /
    # SimpleObjAllocator (ADTObj, NNVM tuples, NDArray pimpls, etc.).  The
    # arithmetic-overflow hardening for those allocations is already in place;
    # demote this specific warning to non-error so release builds compile
    # cleanly while leaving the warning text visible in logs.  See issues.md
    # CN2 for context.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
       CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
      target_compile_options(mxnet PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=array-bounds>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=stringop-overflow>")
    endif()
    if(USE_CUDA)
      # Note: "=" is required to avoid breaking ccache
      string(APPEND CMAKE_CUDA_FLAGS " -Werror=cross-execution-space-call")
    endif()
  endif()
  if(ENABLE_TESTCOVERAGE)
    find_program(GCOV_PATH gcov REQUIRED)
    target_compile_options(mxnet PUBLIC "--coverage")
    target_link_libraries(mxnet PUBLIC gcov)
  endif()
  if(APPLE)
    target_link_options(mxnet PRIVATE
      "LINKER:-exported_symbols_list,${PROJECT_SOURCE_DIR}/cmake/libmxnet.sym")
  else()
    set_target_properties(mxnet PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  endif()
elseif(MSVC)
  if(USE_CUDA)
    if(USE_SPLIT_ARCH_DLL)
      add_executable(gen_warp tools/windowsbuild/gen_warp.cpp)
      add_library(mxnet SHARED tools/windowsbuild/warp_dll.cpp ${CMAKE_BINARY_DIR}/warp_gen_cpp.cpp
                  ${CMAKE_BINARY_DIR}/warp_gen.asm)
      target_link_libraries(mxnet PRIVATE cudart Shlwapi)
      list(GET arch_code_list 0 mxnet_first_arch)
      foreach(arch ${arch_code_list})
        add_library(mxnet_${arch} SHARED ${SOURCE})
        target_link_libraries(mxnet_${arch} PUBLIC mshadow)
        target_compile_definitions(mxnet_${arch} PRIVATE -DWIN32_LEAN_AND_MEAN)
        target_compile_options(
          mxnet_${arch}
          PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_${arch}>"
        )
        target_compile_options(
          mxnet_${arch}
          PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_${arch},compute_${arch}>"
        )
        if(MXNET_FORCE_SHARED_CRT)
          target_compile_options(
            mxnet_${arch} 
            PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MDd -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")          
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:MINSIZEREL>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
        else()
          target_compile_options(
            mxnet_${arch} 
            PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MTd -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")          
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:MINSIZEREL>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
        endif()
      endforeach()

      add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/warp_gen_cpp.cpp ${CMAKE_BINARY_DIR}/warp_gen.asm
        COMMAND gen_warp $<TARGET_FILE:mxnet_${mxnet_first_arch}> WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ DEPENDS $<TARGET_FILE:mxnet_${mxnet_first_arch}>)
    else(USE_SPLIT_ARCH_DLL)
      string(REPLACE ";" " " NVCC_FLAGS_ARCH "${NVCC_FLAGS_ARCH}")
      string(APPEND CMAKE_CUDA_FLAGS " ${CUDA_ARCH_FLAGS_SPACES}")
      add_library(mxnet SHARED ${SOURCE})
      target_compile_definitions(mxnet PRIVATE -DWIN32_LEAN_AND_MEAN)
      target_link_libraries(mxnet PUBLIC mshadow)
      if(MXNET_FORCE_SHARED_CRT)
        target_compile_options(
            mxnet 
            PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MDd -Gy /bigobj>")
        target_compile_options(
            mxnet
            PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
        target_compile_options(
            mxnet
            PRIVATE "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
        target_compile_options(
            mxnet
            PRIVATE "$<$<AND:$<CONFIG:MINSIZEREL>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MD -Gy /bigobj>")
      else()
        target_compile_options(
          mxnet 
          PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MTd -Gy /bigobj>")
        target_compile_options(
          mxnet
          PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")          
        target_compile_options(
          mxnet
          PRIVATE "$<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
        target_compile_options(
          mxnet
          PRIVATE "$<$<AND:$<CONFIG:MINSIZEREL>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
      endif()
    endif(USE_SPLIT_ARCH_DLL)
  else()
    add_library(mxnet SHARED ${SOURCE})
    target_compile_definitions(mxnet PRIVATE -DWIN32_LEAN_AND_MEAN)
    target_link_libraries(mxnet PUBLIC mshadow)
  endif()
endif()
target_compile_definitions(mxnet PUBLIC DMLC_LOG_FATAL_THROW=$<BOOL:${LOG_FATAL_THROW}>)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(mxnet PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>")
endif()

# extension libraries (custom operators, custom subgraphs) are built by default
add_library(customop_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/gemm_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
add_library(transposecsr_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/transposecsr_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
add_library(transposerowsp_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/transposerowsp_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
add_library(subgraph_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_subgraph/subgraph_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
add_library(pass_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_pass/pass_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)

if(IS_DIRECTORY ${BUILD_EXTENSION_PATH})
  if(MSVC)
    message(FATAL_ERROR "Windows builds are not support for external ops")
  else()
    add_subdirectory(${BUILD_EXTENSION_PATH} ${BUILD_EXTENSION_PATH}/build)
  endif()
endif()

target_include_directories(customop_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(transposecsr_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(transposerowsp_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(subgraph_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(pass_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
if(USE_CUDA)
  add_library(customop_gpu_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/relu_lib.cu ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/relu_lib.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_api.cc)
  target_include_directories(customop_gpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op)
endif()
if(UNIX)
  if(USE_CUDA)
    target_compile_options(customop_gpu_lib PUBLIC -shared)
  endif()
elseif(MSVC)
  target_compile_options(customop_lib PUBLIC /LD)
  target_compile_options(transposecsr_lib PUBLIC /LD)
  target_compile_options(transposerowsp_lib PUBLIC /LD)
  target_compile_options(subgraph_lib PUBLIC /LD)
  target_compile_options(pass_lib PUBLIC /LD)
  set_target_properties(customop_lib PROPERTIES PREFIX "lib")
  set_target_properties(transposecsr_lib PROPERTIES PREFIX "lib")
  set_target_properties(transposerowsp_lib PROPERTIES PREFIX "lib")
  set_target_properties(subgraph_lib PROPERTIES PREFIX "lib")
  set_target_properties(pass_lib PROPERTIES PREFIX "lib")
  if(USE_CUDA)
    if(MXNET_FORCE_SHARED_CRT)
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-LD -MD>")
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/LD>")
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/MD>")
    else()
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-LD -MT>")
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/LD>")
      target_compile_options(customop_gpu_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/MT>")
    endif()
    set_target_properties(customop_gpu_lib PROPERTIES PREFIX "lib")
  endif()
endif()

if(USE_DIST_KVSTORE)
  add_subdirectory("3rdparty/ps-lite")
  add_definitions(-DMXNET_USE_DIST_KVSTORE)
  list(APPEND mxnet_LINKER_LIBS pslite)
  set_target_properties(pslite PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
endif()

if(USE_INTGEMM)
  target_link_libraries(mxnet PRIVATE intgemm)
endif()

function(BuildTVMOP)
  # scope the variables in BuildTVM.cmake to avoid conflict
  include(cmake/BuildTVM.cmake)
  add_subdirectory("3rdparty/tvm")
  set_target_properties(tvm PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
  set_target_properties(tvm_runtime PROPERTIES CXX_CLANG_TIDY "")  # don't lint 3rdparty dependency
endfunction()

if(USE_TVM_OP)
  list(APPEND mxnet_LINKER_LIBS tvm_runtime)
  BuildTVMOP()
  find_package(Python3 REQUIRED)
  set(TVM_OP_COMPILE_OPTIONS "-o${CMAKE_CURRENT_BINARY_DIR}" "--config" "${CMAKE_CURRENT_BINARY_DIR}/tvmop.conf" "-L" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm")
  if(UNIX AND NOT APPLE)
    set(LD_LIBRARY_PATH "LD_LIBRARY_PATH")
  elseif(APPLE)
    set(LD_LIBRARY_PATH "DYLD_LIBRARY_PATH")
  endif()
  if(USE_CUDA)
    set(TVM_OP_COMPILE_OPTIONS "${TVM_OP_COMPILE_OPTIONS}" "--cuda-arch" "\"${CUDA_ARCH_FLAGS}\"")
  endif()

  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH="${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/python:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/topi/python:${CMAKE_CURRENT_SOURCE_DIR}/contrib"
    ${LD_LIBRARY_PATH}=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm:$ENV{${LD_LIBRARY_PATH}}
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/contrib/tvmop/compile.py ${TVM_OP_COMPILE_OPTIONS}
  )
endif()

if(MSVC)
  if(USE_SPLIT_ARCH_DLL AND USE_CUDA)
    foreach(arch ${arch_code_list})
      target_link_libraries(mxnet_${arch} PUBLIC ${mxnet_LINKER_LIBS})
      target_link_libraries(mxnet_${arch} PUBLIC dmlc)
    endforeach()
  endif()
endif()

target_link_libraries(mxnet PUBLIC ${mxnet_LINKER_LIBS})
target_link_libraries(mxnet PUBLIC dmlc)

if(USE_OPENCV AND OpenCV_VERSION_MAJOR GREATER 2)
  add_executable(im2rec "tools/im2rec.cc")
  target_link_libraries(im2rec PRIVATE mxnet)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(im2rec PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>")
  endif()
else()
    message(WARNING "OpenCV_VERSION_MAJOR: ${OpenCV_VERSION_MAJOR}, version 3 with imgcodecs \
    is required for im2rec, im2rec will not be available")
endif()


if(MSVC AND USE_MXNET_LIB_NAMING)
  set_target_properties(mxnet PROPERTIES OUTPUT_NAME "libmxnet")
endif()


# NOTE: Public headers will be installed into ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}, see
#       https://cmake.org/cmake/help/v3.13/variable/CMAKE_INSTALL_PREFIX.html
#       https://cmake.org/cmake/help/v3.13/module/GNUInstallDirs.html
install(TARGETS mxnet
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT   MXNET_Runtime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT            MXNET_Runtime
          NAMELINK_COMPONENT   MXNET_Development
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT   MXNET_Development
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mshadow/mshadow/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mshadow)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mxnet)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/nnvm/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(INSTALL_EXAMPLES)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/example  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
endif()

if(USE_SIGNAL_HANDLER)
    add_definitions(-DMXNET_USE_SIGNAL_HANDLER=1)
endif()

# AUTO_INSTALL_DIR -> Optional: specify post-build install direcory
if(AUTO_INSTALL_DIR)
  # ---[ Install Includes
  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${AUTO_INSTALL_DIR}/include
    )

  # ---[ Install Examples
  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/example ${AUTO_INSTALL_DIR}/example
    )
endif()

if(INSTALL_PYTHON_VERSIONS)
  message(STATUS "Installing for python versions: ${INSTALL_PYTHON_VERSIONS}")
  foreach(version ${INSTALL_PYTHON_VERSIONS})
    set(outdir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python${version}/site-packages/mxnet)
    add_custom_command(TARGET mxnet POST_BUILD
      COMMAND mkdir -p ${outdir}
      COMMAND cp -ru ${CMAKE_CURRENT_SOURCE_DIR}/python/mxnet/* ${outdir}
      )
  endforeach()
endif()

if(USE_CPP_PACKAGE)
  add_subdirectory(cpp-package)
  target_compile_definitions(mxnet PUBLIC MXNET_USE_CPP_PACKAGE=1)
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Distribution")
  # Staticbuild applies linker version script to hide private symbols, breaking unit tests
  add_subdirectory(tests)
endif()

# ---[ Linter target
find_package(Python3)
set(LINT_DIRS "include src plugin tests")
set(EXCLUDE_PATH "src/operator/contrib/ctc_include")
add_custom_target(mxnet_lint COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE} -DLINT_DIRS=${LINT_DIRS} -DPROJECT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DPROJECT_NAME=mxnet -DEXCLUDE_PATH=${EXCLUDE_PATH} -P ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/cmake/lint.cmake)

if(BUILD_CYTHON_MODULES)
  include(cmake/BuildCythonModules.cmake)
  add_cython_modules(3) # Build cython module for python3 if python3 is found
  if(NOT ${PYTHON3_FOUND})
    message(FATAL_ERROR "No python interpreter found to build cython modules")
  endif()
endif()

# https://github.com/apache/incubator-mxnet/issues/20145
if(UNIX)
    if(USE_NVML)
      find_package(NVML)
      if(NVML_FOUND)
         target_compile_definitions(mxnet PRIVATE NVML_NO_UNVERSIONED_FUNC_DEFS)
    endif()
  endif()
endif()
