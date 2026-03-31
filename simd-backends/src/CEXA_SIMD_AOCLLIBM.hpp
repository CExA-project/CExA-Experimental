// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

#ifndef CEXA_SIMD_AOCLLIBM_HPP
#define CEXA_SIMD_AOCLLIBM_HPP

#include <Kokkos_Core.hpp>
#define AMD_LIBM_VEC_EXPERIMENTAL
#include <amdlibm_vec.h>
#undef AMD_LIBM_VEC_EXPERIMENTAL

namespace Kokkos {

// NOTE: Most of the commented out functions below are not implemented in AOCL
// LibM. abs, sqrt and copysign are commented out because accelerated versions
// are already available in Kokkos SIMD.

#if defined(KOKKOS_ARCH_AVX2)

#include <Kokkos_SIMD_AVX2.hpp>

#define CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION(func)                   \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<float,                                      \
                               Experimental::simd_abi::avx2_fixed_size<4>> \
      func(Experimental::basic_simd<                                       \
           float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {  \
    return Experimental::basic_simd<                                       \
        float, Experimental::simd_abi::avx2_fixed_size<4>>(                \
        amd_vrs4_##func##f(static_cast<__m128>(a)));                       \
  }

#define CEXA_IMPL_SIMD_AOCL_AVX2_S8_UNARY_FUNCTION(func)                   \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<float,                                      \
                               Experimental::simd_abi::avx2_fixed_size<8>> \
      func(Experimental::basic_simd<                                       \
           float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {  \
    return Experimental::basic_simd<                                       \
        float, Experimental::simd_abi::avx2_fixed_size<8>>(                \
        amd_vrs8_##func##f(static_cast<__m256>(a)));                       \
  }

#define CEXA_IMPL_SIMD_AOCL_AVX2_D4_UNARY_FUNCTION(func)                   \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                      \
      Experimental::basic_simd<double,                                     \
                               Experimental::simd_abi::avx2_fixed_size<4>> \
      func(Experimental::basic_simd<                                       \
           double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) { \
    return Experimental::basic_simd<                                       \
        double, Experimental::simd_abi::avx2_fixed_size<4>>(               \
        amd_vrd4_##func(static_cast<__m256d>(a)));                         \
  }

#define CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(func) \
  CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION(func)    \
  CEXA_IMPL_SIMD_AOCL_AVX2_S8_UNARY_FUNCTION(func)    \
  CEXA_IMPL_SIMD_AOCL_AVX2_D4_UNARY_FUNCTION(func)

// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(abs)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(exp)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(exp2)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(log)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(log10)
CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION(log10)
CEXA_IMPL_SIMD_AOCL_AVX2_S8_UNARY_FUNCTION(log10)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(log2)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(sqrt)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(cbrt)
CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION(cbrt)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(sin)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(cos)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(tan)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(asin)
CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION(asin)
CEXA_IMPL_SIMD_AOCL_AVX2_S8_UNARY_FUNCTION(asin)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(acos)
CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION(acos)
CEXA_IMPL_SIMD_AOCL_AVX2_S8_UNARY_FUNCTION(acos)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(atan)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(sinh)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(cosh)
CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION(cosh)
CEXA_IMPL_SIMD_AOCL_AVX2_S8_UNARY_FUNCTION(cosh)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(tanh)
CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION(tanh)
CEXA_IMPL_SIMD_AOCL_AVX2_S8_UNARY_FUNCTION(tanh)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(asinh)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(acosh)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(atanh)
CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(erf)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(erfc)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(tgamma)
// CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION(lgamma)

#define CEXA_IMPL_SIMD_AOCL_AVX2_BINARY_FUNCTION(func)                         \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                          \
      Experimental::basic_simd<double,                                         \
                               Experimental::simd_abi::avx2_fixed_size<4>>     \
      func(Experimental::basic_simd<                                           \
               double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,   \
           Experimental::basic_simd<                                           \
               double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) { \
    return Experimental::basic_simd<                                           \
        double, Experimental::simd_abi::avx2_fixed_size<4>>(                   \
        amd_vrd4_##func(static_cast<__m256d>(a), static_cast<__m256d>(b)));    \
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
        amd_vrs4_##func##f(static_cast<__m128>(a), static_cast<__m128>(b)));   \
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
        amd_vrs8_##func##f(static_cast<__m256>(a), static_cast<__m256>(b)));   \
  }

CEXA_IMPL_SIMD_AOCL_AVX2_BINARY_FUNCTION(pow)
// CEXA_IMPL_SIMD_AOCL_AVX2_BINARY_FUNCTION(hypot)
// CEXA_IMPL_SIMD_AOCL_AVX2_BINARY_FUNCTION(atan2)
// CEXA_IMPL_SIMD_AOCL_AVX2_BINARY_FUNCTION(copysign)

#elif defined(KOKKOS_ARCH_AVX512XEON)

#include <Kokkos_SIMD_AVX512.hpp>

#define CEXA_IMPL_SIMD_AOCL_AVX512_S8_UNARY_FUNCTION(func)                   \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                        \
      Experimental::basic_simd<float,                                        \
                               Experimental::simd_abi::avx512_fixed_size<8>> \
      func(Experimental::basic_simd<                                         \
           float, Experimental::simd_abi::avx512_fixed_size<8>> const& a) {  \
    return Experimental::basic_simd<                                         \
        float, Experimental::simd_abi::avx512_fixed_size<8>>(                \
        amd_vrs8_##func##f(static_cast<__m256>(a)));                         \
  }

#define CEXA_IMPL_SIMD_AOCL_AVX512_S16_UNARY_FUNCTION(func)                   \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                         \
      Experimental::basic_simd<float,                                         \
                               Experimental::simd_abi::avx512_fixed_size<16>> \
      func(Experimental::basic_simd<                                          \
           float, Experimental::simd_abi::avx512_fixed_size<16>> const& a) {  \
    return Experimental::basic_simd<                                          \
        float, Experimental::simd_abi::avx512_fixed_size<16>>(                \
        amd_vrs16_##func##f(static_cast<__m512>(a)));                         \
  }

#define CEXA_IMPL_SIMD_AOCL_AVX512_D8_UNARY_FUNCTION(func)                   \
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION                        \
      Experimental::basic_simd<double,                                       \
                               Experimental::simd_abi::avx512_fixed_size<8>> \
      func(Experimental::basic_simd<                                         \
           double, Experimental::simd_abi::avx512_fixed_size<8>> const& a) { \
    return Experimental::basic_simd<                                         \
        double, Experimental::simd_abi::avx512_fixed_size<8>>(               \
        amd_vrd8_##func(static_cast<__m512d>(a)));                           \
  }

#define CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(func) \
  CEXA_IMPL_SIMD_AOCL_AVX512_S8_UNARY_FUNCTION(func)    \
  CEXA_IMPL_SIMD_AOCL_AVX512_S16_UNARY_FUNCTION(func)   \
  CEXA_IMPL_SIMD_AOCL_AVX512_D8_UNARY_FUNCTION(func)

// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(abs)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(exp)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(exp2)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(log)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(log10)
CEXA_IMPL_SIMD_AOCL_AVX512_S16_UNARY_FUNCTION(log10)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(log2)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(sqrt)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(cbrt)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(sin)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(cos)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(tan)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(asin)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(acos)
CEXA_IMPL_SIMD_AOCL_AVX512_S16_UNARY_FUNCTION(acos)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(atan)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(sinh)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(cosh)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(tanh)
CEXA_IMPL_SIMD_AOCL_AVX512_S16_UNARY_FUNCTION(tanh)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(asinh)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(acosh)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(atanh)
CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(erf)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(erfc)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(tgamma)
// CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION(lgamma)

#define CEXA_IMPL_SIMD_AOCL_AVX512_BINARY_FUNCTION(func)                       \
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
        amd_vrd8_##func(static_cast<__m512d>(a), static_cast<__m512d>(b)));    \
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
        amd_vrs8_##func##f(static_cast<__m256>(a), static_cast<__m256>(b)));   \
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
        amd_vrs16_##func##f(static_cast<__m512>(a), static_cast<__m512>(b)));  \
  }

CEXA_IMPL_SIMD_AOCL_AVX512_BINARY_FUNCTION(pow)
// CEXA_IMPL_SIMD_AOCL_AVX512_BINARY_FUNCTION(hypot)
// CEXA_IMPL_SIMD_AOCL_AVX512_BINARY_FUNCTION(atan2)
// CEXA_IMPL_SIMD_AOCL_AVX512_BINARY_FUNCTION(copysign)

#endif

}  // namespace Kokkos

#undef CEXA_IMPL_SIMD_AOCL_AVX2_UNARY_FUNCTION
#undef CEXA_IMPL_SIMD_AOCL_AVX2_S4_UNARY_FUNCTION
#undef CEXA_IMPL_SIMD_AOCL_AVX2_S8_UNARY_FUNCTION
#undef CEXA_IMPL_SIMD_AOCL_AVX2_D4_UNARY_FUNCTION
#undef CEXA_IMPL_SIMD_AOCL_AVX2_BINARY_FUNCTION

#undef CEXA_IMPL_SIMD_AOCL_AVX512_UNARY_FUNCTION
#undef CEXA_IMPL_SIMD_AOCL_AVX512_S8_UNARY_FUNCTION
#undef CEXA_IMPL_SIMD_AOCL_AVX512_S16_UNARY_FUNCTION
#undef CEXA_IMPL_SIMD_AOCL_AVX512_D8_UNARY_FUNCTION
#undef CEXA_IMPL_SIMD_AOCL_AVX512_BINARY_FUNCTION

#endif
