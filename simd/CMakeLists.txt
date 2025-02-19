# SPDX-FileCopyrightText: 2025 CExA-project
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.16)

project(cexa-experimental-simd LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")

include(CTest)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(CPM)
set(CPM_USE_LOCAL_PACKAGES ON)

option(SIMD_BUILD_TESTS "Build the tests" ON)
option(SIMD_BUILD_BENCHMARKS "Build the benchmarks" ON)
option(
  SIMD_USE_INTEL_FP_PRECISE
  "Use the flag -fp-model=precise when compiling with an intel compiler in O3"
  OFF
)
cmake_dependent_option(
  SIMD_BUILD_ACCURACY_BENCHMARK
  "Builds the accuracy benchmarks, requires mpfr"
  ON
  "SIMD_BUILD_BENCHMARKS"
  OFF
)

find_package(Kokkos REQUIRED)

add_subdirectory(src)

check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
check_cxx_compiler_flag("-pthread" COMPILER_SUPPORTS_PTHREAD)

if("${BUILD_TESTING}" AND "${SIMD_BUILD_TESTS}")
  add_subdirectory(unit_tests)
endif()

if("${SIMD_BUILD_BENCHMARKS}")
  add_subdirectory(benchmarks)
endif()
