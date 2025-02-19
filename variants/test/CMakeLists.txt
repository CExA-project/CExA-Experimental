# SPDX-FileCopyrightText: 2025 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

cmake_minimum_required(VERSION 3.21.1)

project(Cexa.Variant LANGUAGES CXX)

find_package(Kokkos REQUIRED)

enable_testing()

add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/googletest)

set(TESTS
  assign.copy
  assign.fwd
  assign.move
  ctor.copy
  ctor.default
  ctor.fwd
  ctor.in_place
  ctor.move
  dtor
  fct.ptr
  get
  hash
  intro
  issue
  json
  mod
  relops
  swap
  virtual
  visit)

foreach(test ${TESTS})
  add_executable(${test} ${test}.cpp)
  set_target_properties(${test} PROPERTIES COMPILE_FLAGS "${compile_flags}")
  target_include_directories(${test} PRIVATE ..)
  target_link_libraries(${test} gtest Kokkos::kokkos)
  add_test(${test} ${test} --gtest_color=yes)
endforeach()
