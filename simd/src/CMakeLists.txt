add_library(math-simd INTERFACE AVX2_Math.hpp)
target_link_libraries(math-simd INTERFACE Kokkos::kokkossimd)
target_include_directories(math-simd INTERFACE .)
