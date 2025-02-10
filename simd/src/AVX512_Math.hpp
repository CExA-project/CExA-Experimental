#ifndef CEXA_EXPERIMENTAL_SIMD_AVX512_HPP
#define CEXA_EXPERIMENTAL_SIMD_AVX512_HPP

#include <array>
#include <cmath>
#include <immintrin.h>

#include "Constants.hpp"

namespace Cexa::Experimental::simd::avx512
{

inline __m512d exp8d(__m512d x) {
    using namespace constants::double_precision;

    const __m512d N = _mm512_roundscale_pd(
        _mm512_mul_pd(x, _mm512_set1_pd(INV_L)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    const __m512i Ni = _mm512_cvtpd_epi64(N);
    const __m512i N2i =
        _mm512_and_si512(Ni, _mm512_set1_epi64(31));  // N2i = Ni % 32, N2i >= 0
    const __m512i N1i = Ni - N2i;
    const __m512d N2 = _mm512_cvtepi64_pd(N2i);
    const __m512d N1 = _mm512_cvtepi64_pd(N1i);

    const __m512d vec_L1 = _mm512_set1_pd(L1);
    const __m512d R1 = _mm512_fnmadd_pd(N, vec_L1, x);
    const __m512d R2 = _mm512_fnmadd_pd(N, _mm512_set1_pd(L2), _mm512_set1_pd(0.0));

    const __m512i M = _mm512_srai_epi64(N1i, 5);  // M = N1i / 32

    const __m512d R = _mm512_add_pd(R1, R2);
    __m512d Q = _mm512_fmadd_pd(R, _mm512_set1_pd(A5), _mm512_set1_pd(A4));
    Q = _mm512_fmadd_pd(R, Q, _mm512_set1_pd(A3));
    Q = _mm512_fmadd_pd(R, Q, _mm512_set1_pd(A2));
    Q = _mm512_fmadd_pd(R, Q, _mm512_set1_pd(A1));
    Q = _mm512_mul_pd(R, Q);
    __m512d P = _mm512_add_pd(R1, _mm512_fmadd_pd(R, Q, R2));

    const __m512d S_lead_vec = _mm512_i64gather_pd(N2i, S_lead.data(), 8);
    const __m512d S_trail_vec = _mm512_i64gather_pd(N2i, S_trail.data(), 8);

    const __m512d S = _mm512_add_pd(S_lead_vec, S_trail_vec);

    // compute 2^M
    const __m512i bias = _mm512_set1_epi64(BIAS);
    const __m512i biased_exp = _mm512_add_epi64(M, bias);
    const __m512i one = _mm512_set1_epi64(1);
    const __m512i normal_shift = _mm512_set1_epi64(EXPONENT_SHIFT);
    const __m512i shift = _mm512_min_epi64(
        normal_shift,
        _mm512_sub_epi64(_mm512_add_epi64(normal_shift, biased_exp), one)
    );
    const __mmask8 shift_mask = _mm512_cmpeq_epi64_mask(shift, normal_shift);
    __m512i shifted_value = _mm512_mask_blend_epi64(shift_mask, one, biased_exp);
    const __mmask8 mul_mask = _mm512_cmpeq_epi64_mask(M, _mm512_set1_epi64(BIAS + 1));
    shifted_value = _mm512_mask_sub_epi64(shifted_value, mul_mask, shifted_value, one);
    const __mmask8 large_exponent_mask =
        _mm512_cmpgt_epi64_mask(M, _mm512_set1_epi64(BIAS + 1));
    __m512d two_M = _mm512_castsi512_pd(_mm512_sllv_epi64(shifted_value, shift));
    two_M = _mm512_mask_blend_pd(
        large_exponent_mask,
        two_M,
        _mm512_set1_pd(std::numeric_limits<double>::infinity())
    );

    const __m512d p2 = _mm512_add_pd(S_lead_vec, _mm512_fmadd_pd(S, P, S_trail_vec));

    __m512d res = _mm512_mul_pd(two_M, p2);
    res = _mm512_mask_mul_pd(res, mul_mask, res, _mm512_set1_pd(2.0));

    // special cases
    const __m512d inf = _mm512_set1_pd(std::numeric_limits<double>::infinity());
    const __m512d minus_inf = _mm512_set1_pd(-std::numeric_limits<double>::infinity());

    const __m512d abs_x = _mm512_andnot_pd(_mm512_set1_pd(-0.0), x);
    const __mmask8 abs_x_lt_t2_mask =
        _mm512_cmp_pd_mask(abs_x, _mm512_set1_pd(THRESHOLD_2), _CMP_LT_OQ);

    const __mmask8 x_over_t1_mask =
        _mm512_cmp_pd_mask(x, _mm512_set1_pd(THRESHOLD_1), _CMP_GT_OQ);
    const __mmask8 x_neg_over_t1_mask =
        _mm512_cmp_pd_mask(x, _mm512_set1_pd(-THRESHOLD_1), _CMP_LT_OQ);

    const __mmask8 inf_mask = _mm512_cmp_pd_mask(x, inf, _CMP_EQ_OQ);
    const __mmask8 minus_inf_mask = _mm512_cmp_pd_mask(x, minus_inf, _CMP_EQ_OQ);
    const __mmask8 input_nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_EQ_OQ);

    res = _mm512_mask_blend_pd(inf_mask | x_over_t1_mask, res, inf);
    res = _mm512_mask_blend_pd(
        minus_inf_mask | x_neg_over_t1_mask,
        res,
        _mm512_set1_pd(0.0)
    );
    res = _mm512_mask_blend_pd(
        abs_x_lt_t2_mask,
        res,
        _mm512_add_pd(_mm512_set1_pd(1.0), x)
    );
    res = _mm512_mask_blend_pd(input_nan_mask, x, res);

    return res;
}

inline __m512 exp16f(__m512 x) {
    using namespace constants::simple_precision;

    const __m512 N = _mm512_roundscale_ps(
        _mm512_mul_ps(x, _mm512_set1_ps(INV_L)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    const __m512i Ni = _mm512_cvtps_epi32(N);
    const __m512i N2i =
        _mm512_and_si512(Ni, _mm512_set1_epi32(32 - 1));  // N2i = Ni % 32, N2i >= 0
    const __m512i N1i = Ni - N2i;
    const __m512 N2 = _mm512_cvtepi32_ps(N2i);
    const __m512 N1 = _mm512_cvtepi32_ps(N1i);

    const __m512 abs_N = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), N);
    const __mmask16 n_ge_512 =
        _mm512_cmp_ps_mask(abs_N, _mm512_set1_ps(512), _CMP_GE_OQ);

    const __m512 vec_L1 = _mm512_set1_ps(L1);
    const __m512 R1_ge_512 =
        _mm512_fnmadd_ps(N2, vec_L1, _mm512_fnmadd_ps(N1, vec_L1, x));
    const __m512 R1_lt_512 = _mm512_fnmadd_ps(N, vec_L1, x);
    const __m512 R1 = _mm512_mask_blend_ps(n_ge_512, R1_lt_512, R1_ge_512);
    const __m512 R2 = _mm512_fnmadd_ps(N, _mm512_set1_ps(L2), _mm512_set1_ps(0.0f));

    const __m512i M = _mm512_srai_epi32(N1i, 5);  // M = N1i / 32

    const __m512 R = _mm512_add_ps(R1, R2);
    __m512 Q = _mm512_fmadd_ps(R, _mm512_set1_ps(A2), _mm512_set1_ps(A1));
    Q = _mm512_mul_ps(R, Q);
    __m512 P = _mm512_add_ps(R1, _mm512_fmadd_ps(R, Q, R2));

    const __m512 S_lead_vec = _mm512_i32gather_ps(N2i, S_lead.data(), 4);
    const __m512 S_trail_vec = _mm512_i32gather_ps(N2i, S_trail.data(), 4);

    const __m512 S = _mm512_add_ps(S_lead_vec, S_trail_vec);

    // compute 2^M
    const __m512i bias = _mm512_set1_epi32(BIAS);
    const __m512i biased_exp = _mm512_add_epi32(M, bias);
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i normal_shift = _mm512_set1_epi32(EXPONENT_SHIFT);
    const __m512i shift = _mm512_min_epi32(
        normal_shift,
        _mm512_sub_epi32(_mm512_add_epi32(normal_shift, biased_exp), one)
    );
    const __mmask16 shift_mask = _mm512_cmpeq_epi32_mask(shift, normal_shift);
    __m512i shifted_value = _mm512_mask_blend_epi32(shift_mask, one, biased_exp);
    const __mmask16 mul_mask = _mm512_cmpeq_epi32_mask(M, _mm512_set1_epi32(BIAS + 1));
    shifted_value = _mm512_mask_sub_epi32(shifted_value, mul_mask, shifted_value, one);
    const __mmask16 large_exponent_mask =
        _mm512_cmpgt_epi32_mask(M, _mm512_set1_epi32(BIAS + 1));
    __m512 two_M = _mm512_castsi512_ps(_mm512_sllv_epi32(shifted_value, shift));
    two_M = _mm512_mask_blend_ps(
        large_exponent_mask,
        two_M,
        _mm512_set1_ps(std::numeric_limits<float>::infinity())
    );

    const __m512 p2 = _mm512_add_ps(S_lead_vec, _mm512_fmadd_ps(S, P, S_trail_vec));

    __m512 res = _mm512_mul_ps(two_M, p2);
    res = _mm512_mask_mul_ps(res, mul_mask, res, _mm512_set1_ps(2.0f));

    // special cases
    const __m512 inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    const __m512 minus_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());

    const __m512 abs_x = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), x);
    const __mmask16 abs_x_lt_t2_mask =
        _mm512_cmp_ps_mask(abs_x, _mm512_set1_ps(THRESHOLD_2), _CMP_LT_OQ);

    const __mmask16 x_over_t1_mask =
        _mm512_cmp_ps_mask(x, _mm512_set1_ps(THRESHOLD_1), _CMP_GT_OQ);
    const __mmask16 x_neg_over_t1_mask =
        _mm512_cmp_ps_mask(x, _mm512_set1_ps(-THRESHOLD_1), _CMP_LT_OQ);

    const __mmask16 inf_mask = _mm512_cmp_ps_mask(x, inf, _CMP_EQ_OQ);
    const __mmask16 minus_inf_mask = _mm512_cmp_ps_mask(x, minus_inf, _CMP_EQ_OQ);
    const __mmask16 input_nan_mask = _mm512_cmp_ps_mask(x, x, _CMP_EQ_OQ);

    res = _mm512_mask_blend_ps(inf_mask | x_over_t1_mask, res, inf);
    res = _mm512_mask_blend_ps(
        minus_inf_mask | x_neg_over_t1_mask,
        res,
        _mm512_set1_ps(0.0f)
    );
    res = _mm512_mask_blend_ps(
        abs_x_lt_t2_mask,
        res,
        _mm512_add_ps(_mm512_set1_ps(1.0f), x)
    );
    res = _mm512_mask_blend_ps(input_nan_mask, x, res);

    return res;
}

inline __m256 exp8f(__m256 x) {
    using namespace constants::simple_precision;

    const __m256 N = _mm256_roundscale_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(INV_L)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    const __m256i Ni = _mm256_cvtps_epi32(N);
    const __m256i N2i =
        _mm256_and_si256(Ni, _mm256_set1_epi32(32 - 1));  // N2i = Ni % 32, N2i >= 0
    const __m256i N1i = Ni - N2i;
    const __m256 N2 = _mm256_cvtepi32_ps(N2i);
    const __m256 N1 = _mm256_cvtepi32_ps(N1i);

    const __m256 abs_N = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), N);
    const __mmask8 n_ge_512 = _mm256_cmp_ps_mask(abs_N, _mm256_set1_ps(512), _CMP_GE_OQ);

    const __m256 vec_L1 = _mm256_set1_ps(L1);
    const __m256 R1_ge_512 =
        _mm256_fnmadd_ps(N2, vec_L1, _mm256_fnmadd_ps(N1, vec_L1, x));
    const __m256 R1_lt_512 = _mm256_fnmadd_ps(N, vec_L1, x);
    const __m256 R1 = _mm256_mask_blend_ps(n_ge_512, R1_lt_512, R1_ge_512);
    const __m256 R2 = _mm256_fnmadd_ps(N, _mm256_set1_ps(L2), _mm256_set1_ps(0.0f));

    const __m256i M = _mm256_srai_epi32(N1i, 5);  // M = N1i / 32

    const __m256 R = _mm256_add_ps(R1, R2);
    __m256 Q = _mm256_fmadd_ps(R, _mm256_set1_ps(A2), _mm256_set1_ps(A1));
    Q = _mm256_mul_ps(R, Q);
    __m256 P = _mm256_add_ps(R1, _mm256_fmadd_ps(R, Q, R2));

    const __m256 S_lead_vec = _mm256_i32gather_ps(S_lead.data(), N2i, 4);
    const __m256 S_trail_vec = _mm256_i32gather_ps(S_trail.data(), N2i, 4);

    const __m256 S = _mm256_add_ps(S_lead_vec, S_trail_vec);

    // compute 2^M
    const __m256i bias = _mm256_set1_epi32(BIAS);
    const __m256i biased_exp = _mm256_add_epi32(M, bias);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i normal_shift = _mm256_set1_epi32(EXPONENT_SHIFT);
    const __m256i shift = _mm256_min_epi32(
        normal_shift,
        _mm256_sub_epi32(_mm256_add_epi32(normal_shift, biased_exp), one)
    );
    const __mmask8 shift_mask = _mm256_cmpeq_epi32_mask(shift, normal_shift);
    __m256i shifted_value = _mm256_mask_blend_epi32(shift_mask, one, biased_exp);
    const __mmask8 mul_mask = _mm256_cmpeq_epi32_mask(M, _mm256_set1_epi32(BIAS + 1));
    shifted_value = _mm256_mask_sub_epi32(shifted_value, mul_mask, shifted_value, one);
    const __mmask8 large_exponent_mask =
        _mm256_cmpgt_epi32_mask(M, _mm256_set1_epi32(BIAS + 1));
    __m256 two_M = _mm256_castsi256_ps(_mm256_sllv_epi32(shifted_value, shift));
    two_M = _mm256_mask_blend_ps(
        large_exponent_mask,
        two_M,
        _mm256_set1_ps(std::numeric_limits<float>::infinity())
    );

    const __m256 p2 = _mm256_add_ps(S_lead_vec, _mm256_fmadd_ps(S, P, S_trail_vec));

    __m256 res = _mm256_mul_ps(two_M, p2);
    res = _mm256_mask_mul_ps(res, mul_mask, res, _mm256_set1_ps(2.0f));

    // special cases
    const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    const __m256 minus_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

    const __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    const __mmask8 abs_x_lt_t2_mask =
        _mm256_cmp_ps_mask(abs_x, _mm256_set1_ps(THRESHOLD_2), _CMP_LT_OQ);

    const __mmask8 x_over_t1_mask =
        _mm256_cmp_ps_mask(x, _mm256_set1_ps(THRESHOLD_1), _CMP_GT_OQ);
    const __mmask8 x_neg_over_t1_mask =
        _mm256_cmp_ps_mask(x, _mm256_set1_ps(-THRESHOLD_1), _CMP_LT_OQ);

    const __mmask8 inf_mask = _mm256_cmp_ps_mask(x, inf, _CMP_EQ_OQ);
    const __mmask8 minus_inf_mask = _mm256_cmp_ps_mask(x, minus_inf, _CMP_EQ_OQ);
    const __mmask8 input_nan_mask = _mm256_cmp_ps_mask(x, x, _CMP_EQ_OQ);

    res = _mm256_mask_blend_ps(inf_mask | x_over_t1_mask, res, inf);
    res = _mm256_mask_blend_ps(
        minus_inf_mask | x_neg_over_t1_mask,
        res,
        _mm256_set1_ps(0.0f)
    );
    res = _mm256_mask_blend_ps(
        abs_x_lt_t2_mask,
        res,
        _mm256_add_ps(_mm256_set1_ps(1.0f), x)
    );
    res = _mm256_mask_blend_ps(input_nan_mask, x, res);

    return res;
}

}  // namespace Cexa::Experimental::simd::avx512

#endif
