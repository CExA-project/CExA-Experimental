set(
  SIMD_MATH_SOURCES
  AVX2_Math.hpp
  Kokkos_SIMD_AVX2_Math.hpp
  AVX512_Math.hpp
  Kokkos_SIMD_AVX512_Math.hpp
  Kokkos_SIMD_AVX_Math.hpp
)
add_library(math-simd INTERFACE ${SIMD_MATH_SOURCES})
target_link_libraries(math-simd INTERFACE Kokkos::kokkossimd)
target_include_directories(math-simd INTERFACE .)
