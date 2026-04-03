// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

#ifndef CEXA_SIMD_SVML_HPP
#define CEXA_SIMD_SVML_HPP

#include <Kokkos_Macros.hpp>
#include <immintrin.h>

namespace Kokkos {

// NOTE: If a function is commented out, it means that the accelerated version
// is already available in Kokkos SIMD, either through auto-vectorization or
// call to the associated intrinsic.
// tgamma and lgamma are not offered in this wrapper as the svml version didn't
// offer good performance.

#if defined(KOKKOS_ARCH_AVX2)

#include <Kokkos_SIMD_AVX2.hpp>

#define CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(func)                           \
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

// CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(abs)
// There are already calls to these svml functions in kokkos simd if using an
// intel compiler
#if KOKKOS_VERSION_LESS(5, 0, 0) || !defined(KOKKOS_COMPILER_INTEL_LLVM)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(exp)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(log)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(cbrt)
#endif
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(exp2)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(log10)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(log2)
// CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(sqrt)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(sin)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(cos)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(tan)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(asin)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(acos)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(atan)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(sinh)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(cosh)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(tanh)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(asinh)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(acosh)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(atanh)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(erf)
CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(erfc)
// CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(tgamma)
// CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION(lgamma)

#define CEXA_IMPL_SVML_AVX2_BINARY_FUNCTION(func)                              \
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

CEXA_IMPL_SVML_AVX2_BINARY_FUNCTION(pow)
CEXA_IMPL_SVML_AVX2_BINARY_FUNCTION(hypot)
CEXA_IMPL_SVML_AVX2_BINARY_FUNCTION(atan2)
// CEXA_IMPL_SVML_AVX2_BINARY_FUNCTION(copysign)

#elif defined(KOKKOS_ARCH_AVX512XEON)

#include <Kokkos_SIMD_AVX512.hpp>

#define CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(func)                            \
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

// CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(abs)
// There are already calls to these svml functions in kokkos simd if using an
// intel compiler
#if KOKKOS_VERSION_LESS(5, 0, 0) || !defined(KOKKOS_COMPILER_INTEL_LLVM)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(exp)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(log)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(cbrt)
#endif
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(exp2)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(log10)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(log2)
// CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(sqrt)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(sin)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(cos)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(tan)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(asin)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(acos)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(atan)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(sinh)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(cosh)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(tanh)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(asinh)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(acosh)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(atanh)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(erf)
CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(erfc)
// CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(tgamma)
// CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION(lgamma)

#define CEXA_IMPL_SVML_AVX512_BINARY_FUNCTION(func)                            \
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

CEXA_IMPL_SVML_AVX512_BINARY_FUNCTION(pow)
CEXA_IMPL_SVML_AVX512_BINARY_FUNCTION(hypot)
CEXA_IMPL_SVML_AVX512_BINARY_FUNCTION(atan2)
// CEXA_IMPL_SVML_AVX512_BINARY_FUNCTION(copysign)

#endif

}  // namespace Kokkos

#undef CEXA_IMPL_SVML_AVX2_UNARY_FUNCTION
#undef CEXA_IMPL_SVML_AVX2_BINARY_FUNCTION

#undef CEXA_IMPL_SVML_AVX512_UNARY_FUNCTION
#undef CEXA_IMPL_SVML_AVX512_BINARY_FUNCTION

#endif
