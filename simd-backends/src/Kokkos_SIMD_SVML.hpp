// SPDX-FileCopyrightText: 2025 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#ifndef KOKKOS_SIMD_SVML_HPP
#define KOKKOS_SIMD_SVML_HPP

#include <Kokkos_Macros.hpp>
#include <immintrin.h>

namespace Kokkos {

#if defined(KOKKOS_ARCH_AVX2)

#include <Kokkos_SIMD_AVX2.hpp>

#define KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(func)                         \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<double,                                     \
                               Experimental::simd_abi::avx2_fixed_size<4>> \
      func(Experimental::basic_simd<                                       \
           double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) { \
    return Experimental::basic_simd<                                       \
        double, Experimental::simd_abi::avx2_fixed_size<4>>(               \
        _mm256_##func##_pd(static_cast<__m256d>(a)));                      \
  }                                                                        \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<float,                                      \
                               Experimental::simd_abi::avx2_fixed_size<4>> \
      func(Experimental::basic_simd<                                       \
           float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {  \
    return Experimental::basic_simd<                                       \
        float, Experimental::simd_abi::avx2_fixed_size<4>>(                \
        _mm_##func##_ps(static_cast<__m128>(a)));                          \
  }                                                                        \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<float,                                      \
                               Experimental::simd_abi::avx2_fixed_size<8>> \
      func(Experimental::basic_simd<                                       \
           float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {  \
    return Experimental::basic_simd<                                       \
        float, Experimental::simd_abi::avx2_fixed_size<8>>(                \
        _mm256_##func##_ps(static_cast<__m256>(a)));                       \
  }

// KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(abs)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(exp)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(exp2)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(log)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(log10)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(log2)
// KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(sqrt)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(cbrt)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(sin)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(cos)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(tan)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(asin)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(acos)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(atan)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(sinh)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(cosh)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(tanh)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(asinh)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(acosh)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(atanh)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(erf)
KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(erfc)
// KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(tgamma)
// KOKKOS_IMPL_SVML_AVX2_UNARY_FUNCTION(lgamma)

#define KOKKOS_IMPL_SVML_AVX2_BINARY_FUNCTION(func)                            \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                          \
      Experimental::basic_simd<double,                                         \
                               Experimental::simd_abi::avx2_fixed_size<4>>     \
      func(Experimental::basic_simd<                                           \
               double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,   \
           Experimental::basic_simd<                                           \
               double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) { \
    return Experimental::basic_simd<                                           \
        double, Experimental::simd_abi::avx2_fixed_size<4>>(                   \
        _mm256_##func##_pd(static_cast<__m256d>(a), static_cast<__m256d>(b))); \
  }                                                                            \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                          \
      Experimental::basic_simd<float,                                          \
                               Experimental::simd_abi::avx2_fixed_size<4>>     \
      func(Experimental::basic_simd<                                           \
               float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,    \
           Experimental::basic_simd<                                           \
               float, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {  \
    return Experimental::basic_simd<                                           \
        float, Experimental::simd_abi::avx2_fixed_size<4>>(                    \
        _mm_##func##_ps(static_cast<__m128>(a), static_cast<__m128>(b)));      \
  }                                                                            \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                          \
      Experimental::basic_simd<float,                                          \
                               Experimental::simd_abi::avx2_fixed_size<8>>     \
      func(Experimental::basic_simd<                                           \
               float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,    \
           Experimental::basic_simd<                                           \
               float, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {  \
    return Experimental::basic_simd<                                           \
        float, Experimental::simd_abi::avx2_fixed_size<8>>(                    \
        _mm256_##func##_ps(static_cast<__m256>(a), static_cast<__m256>(b)));   \
  }

KOKKOS_IMPL_SVML_AVX2_BINARY_FUNCTION(pow)
KOKKOS_IMPL_SVML_AVX2_BINARY_FUNCTION(hypot)
KOKKOS_IMPL_SVML_AVX2_BINARY_FUNCTION(atan2)
// KOKKOS_IMPL_SVML_AVX2_BINARY_FUNCTION(copysign)

#elif defined(KOKKOS_ARCH_AVX512XEON)

#include <Kokkos_SIMD_AVX512.hpp>

#define KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(func)                          \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                         \
      Experimental::basic_simd<double,                                        \
                               Experimental::simd_abi::avx512_fixed_size<8>>  \
      func(Experimental::basic_simd<                                          \
           double, Experimental::simd_abi::avx512_fixed_size<8>> const& a) {  \
    return Experimental::basic_simd<                                          \
        double, Experimental::simd_abi::avx512_fixed_size<8>>(                \
        _mm512_##func##_pd(static_cast<__m512d>(a)));                         \
  }                                                                           \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                         \
      Experimental::basic_simd<float,                                         \
                               Experimental::simd_abi::avx512_fixed_size<8>>  \
      func(Experimental::basic_simd<                                          \
           float, Experimental::simd_abi::avx512_fixed_size<8>> const& a) {   \
    return Experimental::basic_simd<                                          \
        float, Experimental::simd_abi::avx512_fixed_size<8>>(                 \
        _mm256_##func##_ps(static_cast<__m256>(a)));                          \
  }                                                                           \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                         \
      Experimental::basic_simd<float,                                         \
                               Experimental::simd_abi::avx512_fixed_size<16>> \
      func(Experimental::basic_simd<                                          \
           float, Experimental::simd_abi::avx512_fixed_size<16>> const& a) {  \
    return Experimental::basic_simd<                                          \
        float, Experimental::simd_abi::avx512_fixed_size<16>>(                \
        _mm512_##func##_ps(static_cast<__m512>(a)));                          \
  }

// KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(abs)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(exp)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(exp2)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(log)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(log10)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(log2)
// KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(sqrt)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(cbrt)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(sin)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(cos)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(tan)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(asin)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(acos)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(atan)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(sinh)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(cosh)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(tanh)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(asinh)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(acosh)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(atanh)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(erf)
KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(erfc)
// KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(tgamma)
// KOKKOS_IMPL_SVML_AVX512_UNARY_FUNCTION(lgamma)

#define KOKKOS_IMPL_SVML_AVX512_BINARY_FUNCTION(func)                          \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                          \
      Experimental::basic_simd<double,                                         \
                               Experimental::simd_abi::avx512_fixed_size<8>>   \
      func(Experimental::basic_simd<                                           \
               double, Experimental::simd_abi::avx512_fixed_size<8>> const& a, \
           Experimental::basic_simd<                                           \
               double, Experimental::simd_abi::avx512_fixed_size<8>> const&    \
               b) {                                                            \
    return Experimental::basic_simd<                                           \
        double, Experimental::simd_abi::avx512_fixed_size<8>>(                 \
        _mm512_##func##_pd(static_cast<__m512d>(a), static_cast<__m512d>(b))); \
  }                                                                            \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                          \
      Experimental::basic_simd<float,                                          \
                               Experimental::simd_abi::avx512_fixed_size<8>>   \
      func(                                                                    \
          Experimental::basic_simd<                                            \
              float, Experimental::simd_abi::avx512_fixed_size<8>> const& a,   \
          Experimental::basic_simd<                                            \
              float, Experimental::simd_abi::avx512_fixed_size<8>> const& b) { \
    return Experimental::basic_simd<                                           \
        float, Experimental::simd_abi::avx512_fixed_size<8>>(                  \
        _mm256_##func##_ps(static_cast<__m256>(a), static_cast<__m256>(b)));   \
  }                                                                            \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                          \
      Experimental::basic_simd<float,                                          \
                               Experimental::simd_abi::avx512_fixed_size<16>>  \
      func(Experimental::basic_simd<                                           \
               float, Experimental::simd_abi::avx512_fixed_size<16>> const& a, \
           Experimental::basic_simd<                                           \
               float, Experimental::simd_abi::avx512_fixed_size<16>> const&    \
               b) {                                                            \
    return Experimental::basic_simd<                                           \
        float, Experimental::simd_abi::avx512_fixed_size<16>>(                 \
        _mm512_##func##_ps(static_cast<__m512>(a), static_cast<__m512>(b)));   \
  }

KOKKOS_IMPL_SVML_AVX512_BINARY_FUNCTION(pow)
KOKKOS_IMPL_SVML_AVX512_BINARY_FUNCTION(hypot)
KOKKOS_IMPL_SVML_AVX512_BINARY_FUNCTION(atan2)
// KOKKOS_IMPL_SVML_AVX512_BINARY_FUNCTION(copysign)

#endif

}  // namespace Kokkos

#endif
