# SPDX-FileCopyrightText: 2025 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

cmake_minimum_required(VERSION 3.21.1)

project(Cexa.Variant LANGUAGES CXX)

list(APPEND TESTS
  fct.ptr
  virtual)

list(APPEND TESTS_MPARK
  assign.copy
  assign.fwd
  assign.move
  ctor.copy
  ctor.default
  ctor.fwd
  ctor.in_place
  ctor.move
  dtor
  get
  hash
  intro
  issue
  json
  mod
  relops
  swap
  visit)

if("${CEXA_BUILD_MPARK_TESTS}")
  list(APPEND TESTS "${TESTS_MPARK}")
endif()

foreach(test ${TESTS})
  if(${test} IN_LIST TESTS_MPARK)
    add_executable(${test} mpark/${test}.cpp)
  else()
    add_executable(${test} ${test}.cpp)
  endif()
  target_link_libraries(${test} gtest Kokkos::kokkos Cexa.Variant)
  add_test(${test} ${test} --gtest_color=yes)
endforeach()

find_package(benchmark QUIET 1.8.3)
if(benchmark_FOUND)
  message(STATUS "Using google benchmark found in ${benchmark_DIR}")
else()
  message(STATUS "No installed google benchmark found, fetching from GitHub")
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF)

  list(APPEND CMAKE_MESSAGE_INDENT "[benchmark] ")
  FetchContent_Declare(
    googlebenchmark
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz
    URL_HASH MD5=7b93dd03670665684f1b2e9b70ad17fe
  )
  FetchContent_MakeAvailable(googlebenchmark)
  list(POP_BACK CMAKE_MESSAGE_INDENT)

  # Suppress clang-tidy diagnostics on code that we do not have control over
  if(CMAKE_CXX_CLANG_TIDY)
    set_target_properties(benchmark PROPERTIES CXX_CLANG_TIDY "")
  endif()

  target_compile_options(benchmark PRIVATE -w)
  target_compile_options(benchmark_main PRIVATE -w)
endif()


add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE benchmark::benchmark Kokkos::kokkos Cexa.Variant)
target_include_directories(bench SYSTEM PRIVATE ${benchmark_SOURCE_DIR}/include)
add_test(NAME bench COMMAND bench --benchmark_counters_tabular=true --benchmark_out=bench_0.json)
