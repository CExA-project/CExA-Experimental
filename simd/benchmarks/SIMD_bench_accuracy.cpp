// SPDX-FileCopyrightText: 2025 CExA-project
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <AVX2_Math.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_SIMD.hpp>
#include <limits>
#include <mpreal.h>

#ifdef KOKKOS_ARCH_AVX512XEON
#include <AVX512_Math.hpp>
#endif

std::uint32_t ulp_distance(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) {
        return 0;
    }

    std::uint32_t ia = Kokkos::bit_cast<std::uint32_t>(a);
    std::uint32_t ib = Kokkos::bit_cast<std::uint32_t>(b);

    return ia > ib ? ia - ib : ib - ia;
}

void bench_accuracy_kokkos() {
    using simd_type = Kokkos::Experimental::simd<float>;
    constexpr std::uint32_t width = simd_type::size();

    std::uint32_t min_ulp = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t max_ulp = std::numeric_limits<std::uint32_t>::min();
    std::uint32_t max_ulp_val = 0;
    std::uint64_t total_ulp = 0;

    alignas(width * sizeof(float)) float values[width];
    for (std::uint32_t i = 0; i < width; i++) {
        values[i] = Kokkos::bit_cast<float>(i);
    }

    simd_type vec;
    simd_type computed;

    for (std::size_t i = 0;
         i <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
         i += width) {
        vec.copy_from(values, Kokkos::Experimental::simd_flag_aligned);
        computed = Kokkos::exp(vec);

        for (std::size_t lane = 0; lane < width; ++lane) {
            float expected = mpfr::exp(values[lane]).toFloat();
            std::uint32_t ulp = ulp_distance(computed[lane], expected);
            // min_ulp = ulp < min_ulp ? ulp : min_ulp;
            // max_ulp_val = ulp > max_ulp ?
            // Kokkos::bit_cast<std::uint32_t>(values[lane])
            //                             : max_ulp_val;
            // max_ulp = ulp > max_ulp ? ulp : max_ulp;
            if (ulp > max_ulp) {
                max_ulp_val = Kokkos::bit_cast<std::uint32_t>(values[lane]);
                max_ulp = ulp;
                std::cout << "New max: " << max_ulp << " at " << max_ulp_val << " ("
                          << values[lane] << ")" << std::endl;
            }
            total_ulp += ulp;
        }

        for (float& value: values) {
            value =
                Kokkos::bit_cast<float>(Kokkos::bit_cast<std::uint32_t>(value) + width);
        }
    }

    double mean_ulp =
        static_cast<double>(total_ulp) / std::numeric_limits<std::uint32_t>::max();
    // std::cout << "Kokkos: " << mean_ulp << ' ' << min_ulp << ' ' << max_ulp << ' '
    //           << max_ulp_val << std::endl;
    std::cout << "Kokkos: " << mean_ulp << ' ' << max_ulp << ' ' << max_ulp_val
              << std::endl;
}

void bench_accuracy_custom_avx2() {
    constexpr std::uint32_t width = 8;

    std::uint32_t min_ulp = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t max_ulp = std::numeric_limits<std::uint32_t>::min();
    std::uint32_t max_ulp_val = 0;
    std::uint64_t total_ulp = 0;

    alignas(width * sizeof(float)) float values[width];
    for (std::uint32_t i = 0; i < width; i++) {
        values[i] = Kokkos::bit_cast<float>(i);
    }

    alignas(width * sizeof(float)) float computed[width];

    __m256 vec;
    __m256 computed_vec;

    for (std::size_t i = 0;
         i <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
         i += width) {
        vec = _mm256_load_ps(values);
        computed_vec = Cexa::Experimental::simd::avx2::exp8f(vec);
        _mm256_store_ps(computed, computed_vec);

        for (std::size_t lane = 0; lane < width; ++lane) {
            float expected = mpfr::exp(values[lane]).toFloat();
            std::uint32_t ulp = ulp_distance(computed[lane], expected);
            // min_ulp = ulp < min_ulp ? ulp : min_ulp;
            // max_ulp_val = ulp > max_ulp ?
            // Kokkos::bit_cast<std::uint32_t>(values[lane])
            //                             : max_ulp_val;
            // max_ulp = ulp > max_ulp ? ulp : max_ulp;
            if (ulp > max_ulp) {
                max_ulp_val = Kokkos::bit_cast<std::uint32_t>(values[lane]);
                max_ulp = ulp;
                std::cout << "New max: " << max_ulp << " at " << max_ulp_val << " ("
                          << values[lane] << ")" << std::endl;
            }
            total_ulp += ulp;
        }

        for (float& value: values) {
            value =
                Kokkos::bit_cast<float>(Kokkos::bit_cast<std::uint32_t>(value) + width);
        }
    }

    double mean_ulp =
        static_cast<double>(total_ulp) / std::numeric_limits<std::uint32_t>::max();
    // std::cout << "AVX2: " << mean_ulp << ' ' << min_ulp << ' ' << max_ulp << ' ' <<
    // max_ulp_val << std::endl;
    std::cout << "AVX2: " << mean_ulp << ' ' << max_ulp << ' ' << max_ulp_val
              << std::endl;
}

#ifdef KOKKOS_ARCH_AVX512XEON

void bench_accuracy_custom_avx512() {
    constexpr std::uint32_t width = 16;

    std::uint32_t min_ulp = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t max_ulp = std::numeric_limits<std::uint32_t>::min();
    std::uint32_t max_ulp_val = 0;
    std::uint64_t total_ulp = 0;

    alignas(width * sizeof(float)) float values[width];
    for (std::uint32_t i = 0; i < width; i++) {
        values[i] = Kokkos::bit_cast<float>(i);
    }

    alignas(width * sizeof(float)) float computed[width];

    __m512 vec;
    __m512 computed_vec;

    for (std::size_t i = 0;
         i <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
         i += width) {
        vec = _mm512_load_ps(values);
        computed_vec = Cexa::Experimental::simd::avx512::exp16f(vec);
        _mm512_store_ps(computed, computed_vec);

        for (std::size_t lane = 0; lane < width; ++lane) {
            float expected = mpfr::exp(values[lane]).toFloat();
            std::uint32_t ulp = ulp_distance(computed[lane], expected);
            // min_ulp = ulp < min_ulp ? ulp : min_ulp;
            // max_ulp_val = ulp > max_ulp ?
            // Kokkos::bit_cast<std::uint32_t>(values[lane])
            //                             : max_ulp_val;
            // max_ulp = ulp > max_ulp ? ulp : max_ulp;
            if (ulp > max_ulp) {
                max_ulp_val = Kokkos::bit_cast<std::uint32_t>(values[lane]);
                max_ulp = ulp;
                std::cout << "New max: " << max_ulp << " at " << max_ulp_val << " ("
                          << values[lane] << ")" << std::endl;
            }
            total_ulp += ulp;
        }

        for (float& value: values) {
            value =
                Kokkos::bit_cast<float>(Kokkos::bit_cast<std::uint32_t>(value) + width);
        }
    }

    double mean_ulp =
        static_cast<double>(total_ulp) / std::numeric_limits<std::uint32_t>::max();
    // std::cout << "AVX512: " << mean_ulp << ' ' << min_ulp << ' ' << max_ulp << ' '
    //           << max_ulp_val << std::endl;
    std::cout << "AVX512: " << mean_ulp << ' ' << max_ulp << ' ' << max_ulp_val
              << std::endl;
}

#endif

int main() {
    mpfr::mpreal::set_default_prec(std::numeric_limits<float>::digits);
    bench_accuracy_kokkos();
    bench_accuracy_custom_avx2();
#ifdef KOKKOS_ARCH_AVX512XEON
    bench_accuracy_custom_avx512();
#endif
}
