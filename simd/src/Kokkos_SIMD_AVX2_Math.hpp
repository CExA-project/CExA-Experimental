#ifndef CEXA_EXPERIMENTAL_KOKKOS_SIMD_AVX2_MATH_HPP
#define CEXA_EXPERIMENTAL_KOKKOS_SIMD_AVX2_MATH_HPP

#include <AVX2.hpp>
#include <Kokkos_SIMD.hpp>

namespace Kokkos
{

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
    exp(Experimental::
            basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>> const& x) {
    return Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>(
        exp4d(static_cast<__m256d>(x))
    );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
    exp(Experimental::
            basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> const& x) {
    return Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>(
        exp8f(static_cast<__m256>(x))
    );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
    exp(Experimental::
            basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> const& x) {
    return Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>(
        exp4f(static_cast<__m128>(x))
    );
}

}  // namespace Kokkos

#endif
