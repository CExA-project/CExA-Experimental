# SPDX-FileCopyrightText: 2025 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception


list(APPEND EXAMPLES
  virtual_fct_port)

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} gtest Kokkos::kokkos Cexa.Variant)
endforeach()
