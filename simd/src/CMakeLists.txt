# SPDX-FileCopyrightText: 2025 CExA-project
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

set(
  SIMD_MATH_SOURCES
  Constants.hpp
  AVX2_Math.hpp
  AVX512_Math.hpp
  Kokkos_SIMD_AVX_Math.hpp
)
add_library(math-simd INTERFACE ${SIMD_MATH_SOURCES})
target_link_libraries(math-simd INTERFACE Kokkos::kokkossimd)
target_include_directories(math-simd INTERFACE .)
