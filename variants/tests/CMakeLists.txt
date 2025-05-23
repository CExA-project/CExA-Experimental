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
