#ifndef CEXA_EXPERIMENTAL_SIMD_AVX2_HPP
#define CEXA_EXPERIMENTAL_SIMD_AVX2_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

namespace avx2
{

template<typename T, std::size_t width>
constexpr auto init_vector(T x) {
    constexpr bool is_m128 = width == 4 && std::is_same_v<T, float>;
    constexpr bool is_m256 = width == 8 && std::is_same_v<T, float>;
    constexpr bool is_m256d = width == 4 && std::is_same_v<T, double>;

    static_assert(
        is_m128 || is_m256 || is_m256d,
        "only floating point vector types are supported"
    );

    if constexpr (is_m128) {
        return __m128{x, x, x, x};
    } else if constexpr (is_m256d) {
        return __m256d{x, x, x, x};
    } else if constexpr (is_m256) {
        return __m256{x, x, x, x, x, x, x, x};
    }
}

template<typename T, std::size_t width>
constexpr std::array taylor_exp_coeffs = {
    init_vector<T, width>(1.0),
    init_vector<T, width>(1.0 / 2),
    init_vector<T, width>(1.0 / 6),
    init_vector<T, width>(1.0 / 24),
    init_vector<T, width>(1.0 / 120),
    init_vector<T, width>(1.0 / 720),
    init_vector<T, width>(1.0 / 5040),
    init_vector<T, width>(1.0 / 40320),
    init_vector<T, width>(1.0 / 362880),
    init_vector<T, width>(1.0 / 3628800),
    init_vector<T, width>(1.0 / 39916800),
    init_vector<T, width>(1.0 / 479001600),
    init_vector<T, width>(1.0 / 6227020800)
};

// convert a packed double to packed i64 (the intrinsic for this is only available for
// avx512). We need to store because msvc doesnt support proper indexing on simd vectors.
inline __m256i cvtpd_epi64(__m256d x) {
    double buf[4];
    _mm256_storeu_pd(buf, x);
    return _mm256_setr_epi64x(
        static_cast<std::int64_t>(buf[0]),
        static_cast<std::int64_t>(buf[1]),
        static_cast<std::int64_t>(buf[2]),
        static_cast<std::int64_t>(buf[3])
    );
}

inline __m256d exp4d(__m256d x) {
    constexpr __m256d ln2 = init_vector<double, 4>(0.693147180559945309417232121458);
    constexpr __m256d inv_ln2 = init_vector<double, 4>(1.44269504088896340735992468100);
    constexpr __m256d half = init_vector<double, 4>(0.5);
    constexpr __m256d zero = init_vector<double, 4>(0.0);

    // Range reduction
    // We express e^x as e^(k * ln(2) + r) = e^(k * ln(2)) * e^r = 2^k * e^r
    // k = floor(x / ln(2) + 1/2)
    const __m256d k = _mm256_floor_pd(_mm256_fmadd_pd(x, inv_ln2, half)
    );  // _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC
    // r = x - k * ln(2)
    const __m256d r = _mm256_fnmadd_pd(k, ln2, x);

    // compute 2^k
    const __m256i bias = _mm256_set1_epi64x(1023ll);
    const __m256i k_int = cvtpd_epi64(k);
    const __m256d two_k =
        _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(k_int, bias), 52));

    // compute the taylor approximation of e^r
    __m256d approx = init_vector<double, 4>(1.0);
    __m256d r_pow = r;
    for (__m256d coeff: taylor_exp_coeffs<double, 4>) {
        approx = _mm256_fmadd_pd(r_pow, coeff, approx);
        r_pow = _mm256_mul_pd(r_pow, r);
    }

    __m256d res = _mm256_mul_pd(two_k, approx);

    // handle special values, e^-inf = 0, e^inf = inf, nans are already correctly handled
    constexpr __m256d minus_inf = init_vector<double, 4>(-INFINITY);
    constexpr __m256d inf = init_vector<double, 4>(INFINITY);

    const __m256d inf_mask = _mm256_cmp_pd(x, inf, 0);
    const __m256d minus_inf_mask = _mm256_cmp_pd(x, minus_inf, 0);

    res = _mm256_blendv_pd(res, zero, minus_inf_mask);
    res = _mm256_blendv_pd(res, inf, inf_mask);

    return res;
}

inline __m256 exp8f(__m256 x) {
    constexpr __m256 ln2 = init_vector<float, 8>(0.693147180559945309417232121458);
    constexpr __m256 inv_ln2 = init_vector<float, 8>(1.44269504088896340735992468100);
    constexpr __m256 half = init_vector<float, 8>(0.5);
    constexpr __m256 zero = init_vector<float, 8>(0.0);

    const __m256 k = _mm256_floor_ps(_mm256_fmadd_ps(x, inv_ln2, half));
    const __m256 r = _mm256_fnmadd_ps(k, ln2, x);

    // compute 2^k
    const __m256i bias = _mm256_set1_epi32(127ll);
    const __m256i k_int = _mm256_cvtps_epi32(k);
    const __m256 two_k =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23));

    // compute the taylor approximation of e^r
    __m256 approx = init_vector<float, 8>(1.0);
    __m256 r_pow = r;
    for (__m256 coeff: taylor_exp_coeffs<float, 8>) {
        approx = _mm256_fmadd_ps(r_pow, coeff, approx);
        r_pow = _mm256_mul_ps(r_pow, r);
    }

    __m256 res = _mm256_mul_ps(two_k, approx);

    // special cases
    constexpr __m256 minus_inf = init_vector<float, 8>(-INFINITY);
    constexpr __m256 inf = init_vector<float, 8>(INFINITY);

    const __m256 inf_mask = _mm256_cmp_ps(x, inf, 0);
    const __m256 minus_inf_mask = _mm256_cmp_ps(x, minus_inf, 0);

    res = _mm256_blendv_ps(res, zero, minus_inf_mask);
    res = _mm256_blendv_ps(res, inf, inf_mask);

    return res;
}

inline __m128 exp4f(__m128 x) {
    constexpr __m128 ln2 = init_vector<float, 4>(0.693147180559945309417232121458);
    constexpr __m128 inv_ln2 = init_vector<float, 4>(1.44269504088896340735992468100);
    constexpr __m128 half = init_vector<float, 4>(0.5);
    constexpr __m128 zero = init_vector<float, 4>(0.0);

    const __m128 k = _mm_floor_ps(_mm_fmadd_ps(x, inv_ln2, half));
    const __m128 r = _mm_fnmadd_ps(k, ln2, x);

    // compute 2^k
    const __m128i bias = _mm_set1_epi32(127ll);
    const __m128i k_int = _mm_cvtps_epi32(k);
    const __m128 two_k =
        _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k_int, bias), 23));

    // compute the taylor approximation of e^r
    __m128 approx = init_vector<float, 4>(1.0);
    __m128 r_pow = r;
    for (__m128 coeff: taylor_exp_coeffs<float, 4>) {
        approx = _mm_fmadd_ps(r_pow, coeff, approx);
        r_pow = _mm_mul_ps(r_pow, r);
    }

    __m128 res = _mm_mul_ps(two_k, approx);

    // special cases
    constexpr __m128 minus_inf = init_vector<float, 4>(-INFINITY);
    constexpr __m128 inf = init_vector<float, 4>(INFINITY);

    const __m128 inf_mask = _mm_cmp_ps(x, inf, 0);
    const __m128 minus_inf_mask = _mm_cmp_ps(x, minus_inf, 0);

    res = _mm_blendv_ps(res, zero, minus_inf_mask);
    res = _mm_blendv_ps(res, inf, inf_mask);

    return res;
}

}  // namespace avx2

#endif
