// SPDX-FileCopyrightText: 2025 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#ifndef CEXA_SIMD_SLEEF_HPP
#define CEXA_SIMD_SLEEF_HPP

#include <Kokkos_Core.hpp>
#include <sleef.h>

namespace Kokkos {

// NOTE: If a function is commented out, it means that the accelerated version
// is already available in Kokkos SIMD, either through auto-vectorization or
// call to the associated intrinsic.

#if defined(KOKKOS_ARCH_AVX2)

#include <Kokkos_SIMD_AVX2.hpp>

#define CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(func, prec)                    \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<double,                                     \
                               Experimental::simd_abi::avx2_fixed_size<4>> \
      func(Experimental::basic_simd<                                       \
           double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) { \
    return Experimental::basic_simd<                                       \
        double, Experimental::simd_abi::avx2_fixed_size<4>>(               \
        Sleef_finz_##func##d4_##prec##avx2(static_cast<__m256d>(a)));      \
  }                                                                        \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<float,                                      \
                               Experimental::simd_abi::avx2_fixed_size<4>> \
      func(Experimental::basic_simd<                                       \
           float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {  \
    return Experimental::basic_simd<                                       \
        float, Experimental::simd_abi::avx2_fixed_size<4>>(                \
        Sleef_finz_##func##f4_##prec##avx2128(static_cast<__m128>(a)));    \
  }                                                                        \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<float,                                      \
                               Experimental::simd_abi::avx2_fixed_size<8>> \
      func(Experimental::basic_simd<                                       \
           float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {  \
    return Experimental::basic_simd<                                       \
        float, Experimental::simd_abi::avx2_fixed_size<8>>(                \
        Sleef_finz_##func##f8_##prec##avx2(static_cast<__m256>(a)));       \
  }

// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(abs, )
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(exp, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(exp2, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(log, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(log10, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(log2, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(sqrt, u05)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(cbrt, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(sin, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(cos, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(tan, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(asin, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(acos, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(atan, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(sinh, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(cosh, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(tanh, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(asinh, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(acosh, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(atanh, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(erf, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(erfc, u15)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(tgamma, u10)
CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(lgamma, u10)

// TODO: functions with lower accuracy, see if they provide a substantial
// speedup compared to the accurate versions
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(exp, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(exp2, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(log, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(log10, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(log2, u35)
// // CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(sqrt, u05)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(cbrt, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(sin, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(cos, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(tan, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(asin, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(acos, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(atan, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(sinh, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(cosh, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(tanh, u35)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(asinh, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(acosh, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(atanh, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(erf, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(erfc, u15)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(tgamma, u10)
// CEXA_IMPL_SLEEF_AVX2_UNARY_FUNCTION(lgamma, u10)

#define CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(func, prec)                       \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                          \
      Experimental::basic_simd<double,                                         \
                               Experimental::simd_abi::avx2_fixed_size<4>>     \
      func(Experimental::basic_simd<                                           \
               double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,   \
           Experimental::basic_simd<                                           \
               double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) { \
    return Experimental::basic_simd<                                           \
        double, Experimental::simd_abi::avx2_fixed_size<4>>(                   \
        Sleef_finz_##func##d4_##prec##avx2(static_cast<__m256d>(a),            \
                                           static_cast<__m256d>(b)));          \
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
        Sleef_finz_##func##f4_##prec##avx2128(static_cast<__m128>(a),          \
                                              static_cast<__m128>(b)));        \
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
        Sleef_finz_##func##f8_##prec##avx2(static_cast<__m256>(a),             \
                                           static_cast<__m256>(b)));           \
  }

CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(pow, u10)
CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(hypot, u05)
CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(atan2, u10)
// CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(copysign, )

// TODO: functions with lower accuracy, see if they provide a substantial
// speedup compared to the accurate versions
// CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(pow, u10)
// CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(hypot, u35)
// CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(atan2, u35)
// // CEXA_IMPL_SLEEF_AVX2_BINARY_FUNCTION(copysign, )

#elif defined(KOKKOS_ARCH_AVX512XEON)

#include <Kokkos_SIMD_AVX512.hpp>

#define CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(func, prec)                     \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                         \
      Experimental::basic_simd<double,                                        \
                               Experimental::simd_abi::avx512_fixed_size<8>>  \
      func(Experimental::basic_simd<                                          \
           double, Experimental::simd_abi::avx512_fixed_size<8>> const& a) {  \
    return Experimental::basic_simd<                                          \
        double, Experimental::simd_abi::avx512_fixed_size<8>>(                \
        Sleef_finz_##func##d8_##prec##avx512f(static_cast<__m512d>(a)));      \
  }                                                                           \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                         \
      Experimental::basic_simd<float,                                         \
                               Experimental::simd_abi::avx512_fixed_size<16>> \
      func(Experimental::basic_simd<                                          \
           float, Experimental::simd_abi::avx512_fixed_size<16>> const& a) {  \
    return Experimental::basic_simd<                                          \
        float, Experimental::simd_abi::avx512_fixed_size<16>>(                \
        Sleef_finz_##func##f16_##prec##avx512f(static_cast<__m512>(a)));      \
  }                                                                           \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                         \
      Experimental::basic_simd<float,                                         \
                               Experimental::simd_abi::avx512_fixed_size<8>>  \
      func(Experimental::basic_simd<                                          \
           float, Experimental::simd_abi::avx512_fixed_size<8>> const& a) {   \
    return Experimental::basic_simd<                                          \
        float, Experimental::simd_abi::avx512_fixed_size<8>>(                 \
        Sleef_finz_##func##f8_##prec##avx2(static_cast<__m256>(a)));          \
  }

// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(abs, )
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(exp, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(exp2, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(log, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(log10, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(log2, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(sqrt, u05)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(cbrt, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(sin, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(cos, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(tan, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(asin, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(acos, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(atan, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(sinh, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(cosh, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(tanh, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(asinh, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(acosh, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(atanh, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(erf, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(erfc, u15)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(tgamma, u10)
CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(lgamma, u10)

// TODO: functions with lower accuracy, see if they provide a substantial
// speedup compared to the accurate versions
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(exp, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(exp2, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(log, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(log10, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(log2, u35)
// // CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(sqrt, u05)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(cbrt, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(sin, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(cos, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(tan, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(asin, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(acos, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(atan, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(sinh, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(cosh, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(tanh, u35)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(asinh, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(acosh, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(atanh, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(erf, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(erfc, u15)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(tgamma, u10)
// CEXA_IMPL_SLEEF_AVX512_UNARY_FUNCTION(lgamma, u10)

#define CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(func, prec)                     \
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
        Sleef_finz_##func##d8_##prec##avx512f(static_cast<__m512d>(a),         \
                                              static_cast<__m512d>(b)));       \
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
        Sleef_finz_##func##f16_##prec##avx512f(static_cast<__m512>(a),         \
                                               static_cast<__m512>(b)));       \
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
        Sleef_finz_##func##f8_##prec##avx2(static_cast<__m256>(a),             \
                                           static_cast<__m256>(b)));           \
  }

CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(pow, u10)
CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(hypot, u05)
CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(atan2, u10)
// CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(copysign, )

// TODO: functions with lower accuracy, see if they provide a substantial
// speedup compared to the accurate versions
// CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(pow, u10)
// CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(hypot, u35)
// CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(atan2, u35)
// // CEXA_IMPL_SLEEF_AVX512_BINARY_FUNCTION(copysign, )

#endif

}  // namespace Kokkos

#endif
