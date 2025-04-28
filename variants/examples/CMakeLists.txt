# SPDX-FileCopyrightText: 2025 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

cmake_minimum_required(VERSION 3.21.1)

project(Cexa.Variant LANGUAGES CXX)

list(APPEND EXAMPLES
  virtual_fct_port)

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES COMPILE_FLAGS "${compile_flags}")
  target_link_libraries(${example} gtest Kokkos::kokkos Cexa.Variant)
endforeach()
