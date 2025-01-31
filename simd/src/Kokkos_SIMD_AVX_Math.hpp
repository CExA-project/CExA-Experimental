#ifndef CEXA_EXPERIMENTAL_KOKKOS_SIMD_AVX_MATH_HPP
#define CEXA_EXPERIMENTAL_KOKKOS_SIMD_AVX_MATH_HPP

#ifdef KOKKOS_ARCH_AVX2
#include <Kokkos_SIMD_AVX2_Math.hpp>
#endif

#ifdef KOKKOS_ARCH_AVX512XEON
#include <Kokkos_SIMD_AVX512_Math.hpp>
#endif

#endif
