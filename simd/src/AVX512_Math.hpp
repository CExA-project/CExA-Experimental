#ifndef CEXA_EXPERIMENTAL_SIMD_AVX512_HPP
#define CEXA_EXPERIMENTAL_SIMD_AVX512_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <type_traits>

namespace avx512
{

template<typename T, std::size_t width>
constexpr auto init_vector(T x) {
    constexpr bool is_m256 = width == 8 && std::is_same_v<T, float>;
    constexpr bool is_m512 = width == 16 && std::is_same_v<T, float>;
    constexpr bool is_m512d = width == 8 && std::is_same_v<T, double>;

    static_assert(
        is_m256 || is_m512 || is_m512d,
        "only floating point vector types are supported"
    );

    if constexpr (is_m256) {
        return __m256{x, x, x, x, x, x, x, x};
    } else if constexpr (is_m512d) {
        return __m512d{x, x, x, x, x, x, x, x};
    } else if constexpr (is_m512) {
        return __m512{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
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

inline __m512d exp8d(__m512d x) {
    constexpr __m512d ln2 = init_vector<double, 8>(0.693147180559945309417232121458);
    constexpr __m512d inv_ln2 = init_vector<double, 8>(1.44269504088896340735992468100);
    constexpr __m512d half = init_vector<double, 8>(0.5);
    constexpr __m512d zero = init_vector<double, 8>(0.0);

    // Range reduction
    // We express e^x as e^(k * ln(2) + r) = e^(k * ln(2)) * e^r = 2^k * e^r
    // k = floor(x / ln(2) + 1/2)
    const __m512d k = _mm512_floor_pd(_mm512_fmadd_pd(x, inv_ln2, half)
    );  // _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC
    // r = x - k * ln(2)
    const __m512d r = _mm512_fnmadd_pd(k, ln2, x);

    // compute 2^k
    const __m512i bias = _mm512_set1_epi64(1023ll);
    const __m512i k_int = _mm512_cvtpd_epi64(k);
    const __m512d two_k =
        _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_add_epi64(k_int, bias), 52));

    // compute the taylor approximation of e^r
    __m512d approx = init_vector<double, 8>(1.0);
    __m512d r_pow = r;
    for (__m512d coeff: taylor_exp_coeffs<double, 8>) {
        approx = _mm512_fmadd_pd(r_pow, coeff, approx);
        r_pow = _mm512_mul_pd(r_pow, r);
    }

    __m512d res = _mm512_mul_pd(two_k, approx);

    // handle special values, e^-inf = 0, e^inf = inf, nans are already correctly handled
    constexpr __m512d inf = init_vector<double, 8>(INFINITY);
    constexpr __m512d minus_inf = init_vector<double, 8>(-INFINITY);

    const __mmask8 inf_mask = _mm512_cmp_pd_mask(x, inf, 0);
    const __mmask8 minus_inf_mask = _mm512_cmp_pd_mask(x, minus_inf, 0);

    res = _mm512_mask_blend_pd(inf_mask, res, inf);
    res = _mm512_mask_blend_pd(minus_inf_mask, res, zero);

    return res;
}

inline __m512 exp16f(__m512 x) {
    constexpr __m512 ln2 = init_vector<float, 16>(0.693147180559945309417232121458);
    constexpr __m512 inv_ln2 = init_vector<float, 16>(1.44269504088896340735992468100);
    constexpr __m512 half = init_vector<float, 16>(0.5);
    constexpr __m512 zero = init_vector<float, 16>(0.0);

    const __m512 k = _mm512_floor_ps(_mm512_fmadd_ps(x, inv_ln2, half));
    const __m512 r = _mm512_fnmadd_ps(k, ln2, x);

    // compute 2^k
    const __m512i bias = _mm512_set1_epi32(127ll);
    const __m512i k_int = _mm512_cvtps_epi32(k);
    const __m512 two_k =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(k_int, bias), 23));

    // compute the taylor approximation of e^r
    __m512 approx = init_vector<float, 16>(1.0);
    __m512 r_pow = r;
    for (__m512 coeff: taylor_exp_coeffs<float, 16>) {
        approx = _mm512_fmadd_ps(r_pow, coeff, approx);
        r_pow = _mm512_mul_ps(r_pow, r);
    }

    __m512 res = _mm512_mul_ps(two_k, approx);

    // special cases
    constexpr __m512 inf = init_vector<float, 16>(INFINITY);
    constexpr __m512 minus_inf = init_vector<float, 16>(-INFINITY);

    const __mmask16 inf_mask = _mm512_cmp_ps_mask(x, inf, 0);
    const __mmask16 minus_inf_mask = _mm512_cmp_ps_mask(x, minus_inf, 0);

    res = _mm512_mask_blend_ps(inf_mask, res, inf);
    res = _mm512_mask_blend_ps(minus_inf_mask, res, zero);

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
    constexpr __m256 inf = init_vector<float, 8>(INFINITY);
    constexpr __m256 minus_inf = init_vector<float, 8>(-INFINITY);

    const __mmask8 inf_mask = _mm256_cmp_ps_mask(x, inf, 0);
    const __mmask8 minus_inf_mask = _mm256_cmp_ps_mask(x, minus_inf, 0);

    res = _mm256_mask_blend_ps(inf_mask, res, inf);
    res = _mm256_mask_blend_ps(minus_inf_mask, res, zero);

    return res;
}

}  // namespace avx512

#endif
