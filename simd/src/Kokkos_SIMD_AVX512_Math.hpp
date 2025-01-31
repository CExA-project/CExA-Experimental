#ifndef CEXA_EXPERIMENTAL_KOKKOS_SIMD_AVX512_MATH_HPP
#define CEXA_EXPERIMENTAL_KOKKOS_SIMD_AVX512_MATH_HPP

#include <AVX512_Math.hpp>
#include <Kokkos_SIMD.hpp>

namespace Kokkos
{

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::basic_simd<double, Experimental::simd_abi::avx512_fixed_size<8>>
    exp(Experimental::
            basic_simd<double, Experimental::simd_abi::avx512_fixed_size<8>> const& x) {
    return Experimental::
        basic_simd<double, Experimental::simd_abi::avx512_fixed_size<8>>(
            avx512::exp8d(static_cast<__m512d>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::basic_simd<float, Experimental::simd_abi::avx512_fixed_size<16>>
    exp(Experimental::
            basic_simd<float, Experimental::simd_abi::avx512_fixed_size<16>> const& x) {
    return Experimental::
        basic_simd<float, Experimental::simd_abi::avx512_fixed_size<16>>(
            avx512::exp16f(static_cast<__m512>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::basic_simd<float, Experimental::simd_abi::avx512_fixed_size<8>>
    exp(Experimental::
            basic_simd<float, Experimental::simd_abi::avx512_fixed_size<8>> const& x) {
    return Experimental::basic_simd<float, Experimental::simd_abi::avx512_fixed_size<8>>(
        avx512::exp8f(static_cast<__m256>(x))
    );
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4

namespace Experimental
{

[[nodiscard]] KOKKOS_DEPRECATED KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx512_fixed_size<8>>
    exp(basic_simd<double, simd_abi::avx512_fixed_size<8>> const& x) {
    return Kokkos::exp(x);
}

[[nodiscard]] KOKKOS_DEPRECATED KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx512_fixed_size<16>>
    exp(basic_simd<float, simd_abi::avx512_fixed_size<16>> const& x) {
    return Kokkos::exp(x);
}

[[nodiscard]] KOKKOS_DEPRECATED KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx512_fixed_size<8>>
    exp(basic_simd<float, simd_abi::avx512_fixed_size<8>> const& x) {
    return Kokkos::exp(x);
}

}  // namespace Experimental

#endif

}  // namespace Kokkos

#endif
