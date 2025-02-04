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

    const __m512d k = _mm512_floor_pd(_mm512_fmadd_pd(x, inv_ln2, half));
    const __m512d r = _mm512_fnmadd_pd(k, ln2, x);

    // compute 2^k
    const __m512i bias = _mm512_set1_epi64(1023ll);
    const __m512i k_int = _mm512_cvtpd_epi64(k);
    const __m512i biased_exp = _mm512_add_epi64(k_int, bias);

    const __m512i one = _mm512_set1_epi64(1);
    const __m512i normal_shift = _mm512_set1_epi64(52ll);
    // if the biased exponent b is less than 0, 2^k is 1 << (normal_shift - 1 + b)
    const __m512i shift = _mm512_min_epi64(
        normal_shift,
        _mm512_sub_epi64(_mm512_add_epi64(normal_shift, biased_exp), one)
    );

    const __mmask8 shift_mask = _mm512_cmpeq_epi64_mask(shift, normal_shift);
    __m512i shifted_value = _mm512_mask_blend_epi64(shift_mask, one, biased_exp);
    const __mmask8 mul_mask = _mm512_cmpeq_epi64_mask(k_int, _mm512_set1_epi64(1024ll));
    shifted_value = _mm512_mask_sub_epi64(shifted_value, mul_mask, shifted_value, one);
    const __m512d two_k = _mm512_castsi512_pd(_mm512_sllv_epi64(shifted_value, shift));

    // compute the taylor approximation of e^r
    __m512d approx = init_vector<double, 8>(1.0);
    __m512d r_pow = r;
    for (__m512d coeff: taylor_exp_coeffs<double, 8>) {
        approx = _mm512_fmadd_pd(r_pow, coeff, approx);
        r_pow = _mm512_mul_pd(r_pow, r);
    }

    __m512d res = _mm512_mul_pd(two_k, approx);
    res = _mm512_mask_mul_pd(res, mul_mask, res, init_vector<double, 8>(2));

    constexpr __m512d minus_inf = init_vector<double, 8>(-INFINITY);
    constexpr __m512d inf = init_vector<double, 8>(INFINITY);
    constexpr __m512d minus_zero = init_vector<double, 8>(-0.0);

    const __mmask8 nan_mask = _mm512_cmp_pd_mask(res, res, _CMP_EQ_OQ);
    const __mmask8 neg_mask =
        _mm512_cmp_pd_mask(_mm512_and_pd(res, minus_zero), minus_zero, _CMP_EQ_OQ);
    const __mmask8 minus_nan_mask = ~nan_mask & neg_mask;
    const __mmask8 input_nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_EQ_OQ);
    const __mmask8 exponent_mask =
        _mm512_cmp_pd_mask(k, _mm512_set1_pd(1024.0), _CMP_GT_OQ);
    const __mmask8 inf_mask = _mm512_cmp_pd_mask(x, inf, _CMP_EQ_OQ);
    const __mmask8 minus_zero_mask = _mm512_cmp_pd_mask(res, minus_zero, _CMP_EQ_OQ);
    const __mmask8 minus_inf_mask = _mm512_cmp_pd_mask(x, minus_inf, _CMP_EQ_OQ);

    res = _mm512_mask_blend_pd(minus_nan_mask, res, zero);
    res = _mm512_mask_blend_pd(input_nan_mask, x, res);
    res = _mm512_mask_blend_pd(minus_inf_mask | minus_zero_mask, res, zero);
    res = _mm512_mask_blend_pd(inf_mask | exponent_mask, res, inf);

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
    const __m512i bias = _mm512_set1_epi32(127);
    const __m512i k_int = _mm512_cvtps_epi32(k);
    const __m512i biased_exp = _mm512_add_epi32(k_int, bias);

    const __m512i one = _mm512_set1_epi32(1);
    const __m512i normal_shift = _mm512_set1_epi32(23);
    const __m512i shift = _mm512_min_epi32(
        normal_shift,
        _mm512_sub_epi32(_mm512_add_epi32(normal_shift, biased_exp), one)
    );

    const __mmask16 shift_mask = _mm512_cmpeq_epi32_mask(shift, normal_shift);
    __m512i shifted_value = _mm512_mask_blend_epi32(shift_mask, one, biased_exp);
    const __mmask16 mul_mask = _mm512_cmpeq_epi32_mask(k_int, _mm512_set1_epi32(128));
    shifted_value = _mm512_mask_sub_epi32(shifted_value, mul_mask, shifted_value, one);
    const __m512 two_k = _mm512_castsi512_ps(_mm512_sllv_epi32(shifted_value, shift));

    // compute the taylor approximation of e^r
    __m512 approx = init_vector<float, 16>(1.0);
    __m512 r_pow = r;
    for (__m512 coeff: taylor_exp_coeffs<float, 16>) {
        approx = _mm512_fmadd_ps(r_pow, coeff, approx);
        r_pow = _mm512_mul_ps(r_pow, r);
    }

    __m512 res = _mm512_mul_ps(two_k, approx);
    res = _mm512_mask_mul_ps(res, mul_mask, res, init_vector<float, 16>(2));

    constexpr __m512 minus_inf = init_vector<float, 16>(-INFINITY);
    constexpr __m512 inf = init_vector<float, 16>(INFINITY);
    constexpr __m512 minus_zero = init_vector<float, 16>(-0.f);

    const __mmask16 nan_mask = _mm512_cmp_ps_mask(res, res, _CMP_EQ_OQ);
    const __mmask16 neg_mask =
        _mm512_cmp_ps_mask(_mm512_and_ps(res, minus_zero), minus_zero, _CMP_EQ_OQ);
    const __mmask16 minus_nan_mask = ~nan_mask & neg_mask;
    const __mmask16 input_nan_mask = _mm512_cmp_ps_mask(x, x, _CMP_EQ_OQ);
    const __mmask16 exponent_mask =
        _mm512_cmp_ps_mask(k, _mm512_set1_ps(128.f), _CMP_GT_OQ);
    const __mmask16 inf_mask = _mm512_cmp_ps_mask(x, inf, _CMP_EQ_OQ);
    const __mmask16 minus_zero_mask = _mm512_cmp_ps_mask(res, minus_zero, _CMP_EQ_OQ);
    const __mmask16 minus_inf_mask = _mm512_cmp_ps_mask(x, minus_inf, _CMP_EQ_OQ);

    res = _mm512_mask_blend_ps(minus_nan_mask, res, zero);
    res = _mm512_mask_blend_ps(input_nan_mask, x, res);
    res = _mm512_mask_blend_ps(minus_inf_mask | minus_zero_mask, res, zero);
    res = _mm512_mask_blend_ps(inf_mask | exponent_mask, res, inf);

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
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256i k_int = _mm256_cvtps_epi32(k);
    const __m256i biased_exp = _mm256_add_epi32(k_int, bias);

    const __m256i one = _mm256_set1_epi32(1);
    const __m256i normal_shift = _mm256_set1_epi32(23);
    const __m256i shift = _mm256_min_epi32(
        normal_shift,
        _mm256_sub_epi32(_mm256_add_epi32(normal_shift, biased_exp), one)
    );

    const __mmask8 shift_mask = _mm256_cmpeq_epi32_mask(shift, normal_shift);
    __m256i shifted_value = _mm256_mask_blend_epi32(shift_mask, one, biased_exp);
    const __mmask8 mul_mask = _mm256_cmpeq_epi32_mask(k_int, _mm256_set1_epi32(128));
    shifted_value = _mm256_mask_sub_epi32(shifted_value, mul_mask, shifted_value, one);
    const __m256 two_k = _mm256_castsi256_ps(_mm256_sllv_epi32(shifted_value, shift));

    // compute the taylor approximation of e^r
    __m256 approx = init_vector<float, 8>(1.0);
    __m256 r_pow = r;
    for (__m256 coeff: taylor_exp_coeffs<float, 8>) {
        approx = _mm256_fmadd_ps(r_pow, coeff, approx);
        r_pow = _mm256_mul_ps(r_pow, r);
    }

    __m256 res = _mm256_mul_ps(two_k, approx);
    res = _mm256_mask_mul_ps(res, mul_mask, res, init_vector<float, 8>(2.0));

    constexpr __m256 minus_inf = init_vector<float, 8>(-INFINITY);
    constexpr __m256 inf = init_vector<float, 8>(INFINITY);
    constexpr __m256 minus_zero = init_vector<float, 8>(-0.0);

    const __mmask8 nan_mask = _mm256_cmp_ps_mask(res, res, _CMP_EQ_OQ);
    const __mmask8 neg_mask =
        _mm256_cmp_ps_mask(_mm256_and_ps(res, minus_zero), minus_zero, _CMP_EQ_OQ);
    const __mmask8 minus_nan_mask = ~nan_mask & neg_mask;
    const __mmask8 input_nan_mask = _mm256_cmp_ps_mask(x, x, _CMP_EQ_OQ);
    const __mmask8 exponent_mask =
        _mm256_cmp_ps_mask(k, _mm256_set1_ps(128.f), _CMP_GT_OQ);
    const __mmask8 inf_mask = _mm256_cmp_ps_mask(x, inf, _CMP_EQ_OQ);
    const __mmask8 minus_zero_mask = _mm256_cmp_ps_mask(res, minus_zero, _CMP_EQ_OQ);
    const __mmask8 minus_inf_mask = _mm256_cmp_ps_mask(x, minus_inf, _CMP_EQ_OQ);

    res = _mm256_mask_blend_ps(minus_nan_mask, res, zero);
    res = _mm256_mask_blend_ps(input_nan_mask, x, res);
    res = _mm256_mask_blend_ps(minus_inf_mask | minus_zero_mask, res, zero);
    res = _mm256_mask_blend_ps(inf_mask | exponent_mask, res, inf);

    return res;
}

}  // namespace avx512

#endif
