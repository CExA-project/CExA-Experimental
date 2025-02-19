// SPDX-FileCopyrightText: 2025 CExA-project
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef CEXA_EXPERIMENTAL_SIMD_TESTING_OPS_HPP
#define CEXA_EXPERIMENTAL_SIMD_TESTING_OPS_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <type_traits>

#include <SIMDTesting_Utilities.hpp>

#ifdef KOKKOS_ARCH_AVX2
#include "AVX2_Math.hpp"

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<double, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>
    custom_exp(Kokkos::Experimental::basic_simd<
               double,
               Kokkos::Experimental::simd_abi::avx2_fixed_size<4>> const& x) {
  return Kokkos::Experimental::basic_simd<
      double, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>(
      Cexa::Experimental::simd::avx2::exp4d(static_cast<__m256d>(x)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<8>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float, Kokkos::Experimental::simd_abi::avx2_fixed_size<8>> const&
                   x) {
  return Kokkos::Experimental::basic_simd<
      float, Kokkos::Experimental::simd_abi::avx2_fixed_size<8>>(
      Cexa::Experimental::simd::avx2::exp8f(static_cast<__m256>(x)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>> const&
                   x) {
  return Kokkos::Experimental::basic_simd<
      float, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>(
      Cexa::Experimental::simd::avx2::exp4f(static_cast<__m128>(x)));
}
#endif

#ifdef KOKKOS_ARCH_AVX512XEON
#include "AVX512_Math.hpp"

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<double, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>
    custom_exp(Kokkos::Experimental::basic_simd<
               double,
               Kokkos::Experimental::simd_abi::avx512_fixed_size<8>> const& x) {
  return Kokkos::Experimental::basic_simd<
      double, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>(
      Cexa::Experimental::simd::avx512::exp8d(static_cast<__m512d>(x)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>>
    custom_exp(
        Kokkos::Experimental::basic_simd<
            float, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>> const&
            x) {
  return Kokkos::Experimental::basic_simd<
      float, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>>(
      Cexa::Experimental::simd::avx512::exp16f(static_cast<__m512>(x)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx512_fixed_size<8>> const& x) {
  return Kokkos::Experimental::basic_simd<
      float, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>(
      Cexa::Experimental::simd::avx512::exp8f(static_cast<__m256>(x)));
}
#endif

template <typename T>
auto custom_exp(T const& a) {
  return Kokkos::exp(a);
}

class exp_op {
 public:
  template <typename T>
  auto on_host(T const& a) const {
    return custom_exp(a);
  }

  template <typename T>
  auto on_host_serial(T const& a) const {
    return Kokkos::exp(a);
  }

  template <bool check_relative_error, class Abi, class Loader, typename T>
  void check_special_values() {
    if constexpr (!std::is_integral_v<T>) {
      using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
      constexpr std::size_t width = simd_type::size();

      gtest_checker checker;
      simd_type computed;
      T computed_serial;

      // nan
      simd_type nan(std::numeric_limits<T>::quiet_NaN());
      computed = on_host(nan);
      for (std::size_t lane = 0; lane < width; lane++) {
        checker.truth(is_nan(computed[lane]));
      }

      T tested_values[] = {0.0,
                           std::numeric_limits<T>::infinity(),
                           -std::numeric_limits<T>::infinity(),
                           -103,
                           88.7,
                           -9.30327e+07,
                           -2.38164398e+10,
                           2.38164398e+10};

      load_masked loader;
      for (std::size_t i = 0; i < std::size(tested_values); i += width) {
        std::size_t nlanes = std::min(width, std::size(tested_values) - i);
        simd_type vec;
        loader.host_load(tested_values + i, nlanes, vec);
        computed = on_host(vec);

        for (std::size_t j = 0; j < nlanes; j++) {
          computed_serial = on_host_serial(tested_values[i + j]);
          if constexpr (check_relative_error) {
            checker.closeness(computed_serial, computed[j]);
          } else {
            checker.equality(computed_serial, computed[j]);
          }
        }
      }
    }
  }
};

#endif
