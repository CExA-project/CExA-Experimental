#ifndef CEXA_EXPERIMENTAL_SIMD_AVX2_HPP
#define CEXA_EXPERIMENTAL_SIMD_AVX2_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <Kokkos_BitManipulation.hpp>

#include "Constants.hpp"

namespace avx2
{

// Convert a packed double to packed i64 (the intrinsic for this is only available for
// avx512). We need to store because msvc doesn't support proper indexing on simd
// vectors.
inline __m256i cvtpd_epi64(__m256d x) {
    alignas(__m256d) double buf[4];
    _mm256_store_pd(buf, x);
    return _mm256_setr_epi64x(
        static_cast<std::int64_t>(buf[0]),
        static_cast<std::int64_t>(buf[1]),
        static_cast<std::int64_t>(buf[2]),
        static_cast<std::int64_t>(buf[3])
    );
}

// Convert a packed i64 to packed double.
inline __m256d cvtepi64_pd(__m256i x) {
    alignas(__m256i) std::int64_t buf[4];
    std::memcpy(buf, &x, sizeof(x));
    return _mm256_setr_pd(
        static_cast<double>(buf[0]),
        static_cast<double>(buf[1]),
        static_cast<double>(buf[2]),
        static_cast<double>(buf[3])
    );
}

// Shifts a vector of 4xi64 by `count` (`count` < 32).
inline __m256i srai_epi64(__m256i x, int count) {
    __m256i sign_mask = _mm256_and_si256(x, _mm256_set1_epi64x(0x8000000000000000));
    sign_mask = _mm256_srai_epi32(sign_mask, count);

    return _mm256_or_si256(_mm256_srli_epi64(x, count), sign_mask);
}

// computes the min of 64 bit integers contained in two vectors
inline __m256i min_epi64(__m256i a, __m256i b) {
    __m256i mask = _mm256_cmpgt_epi64(b, a);
    return _mm256_blendv_epi8(b, a, mask);
}

// Implementation of the exponential from the article :
// Ping-Tak Peter Tang. 1989. Table-driven implementation of the exponential function in
// IEEE floating-point arithmetic. ACM Trans. Math. Softw. 15, 2 (June 1989), 144–157.
// https://doi.org/10.1145/63522.214389
inline __m256d exp4d(__m256d x) {
    using namespace constants::double_precision;

    const __m256d N = _mm256_round_pd(
        _mm256_mul_pd(x, _mm256_set1_pd(INV_L)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    const __m256i Ni = cvtpd_epi64(N);
    const __m256i N2i =
        _mm256_and_si256(Ni, _mm256_set1_epi64x(31));  // N2i = Ni % 32, N2i >= 0
    const __m256i N1i = Ni - N2i;
    const __m256d N2 = cvtepi64_pd(N2i);
    const __m256d N1 = cvtepi64_pd(N1i);

    const __m256d vec_L1 = _mm256_set1_pd(L1);
    const __m256d R1 = _mm256_fnmadd_pd(N, vec_L1, x);
    const __m256d R2 = _mm256_fnmadd_pd(N, _mm256_set1_pd(L2), _mm256_set1_pd(0.0));

    const __m256i M = srai_epi64(N1i, 5);  // M = N1i / 32

    const __m256d R = _mm256_add_pd(R1, R2);
    __m256d Q = _mm256_fmadd_pd(R, _mm256_set1_pd(A5), _mm256_set1_pd(A4));
    Q = _mm256_fmadd_pd(R, Q, _mm256_set1_pd(A3));
    Q = _mm256_fmadd_pd(R, Q, _mm256_set1_pd(A2));
    Q = _mm256_fmadd_pd(R, Q, _mm256_set1_pd(A1));
    Q = _mm256_mul_pd(R, Q);
    __m256d P = _mm256_add_pd(R1, _mm256_fmadd_pd(R, Q, R2));

    const __m256d S_lead_vec = _mm256_i64gather_pd(S_lead.data(), N2i, 8);
    const __m256d S_trail_vec = _mm256_i64gather_pd(S_trail.data(), N2i, 8);

    const __m256d S = _mm256_add_pd(S_lead_vec, S_trail_vec);

    // compute 2^M
    const __m256i bias = _mm256_set1_epi64x(BIAS);
    const __m256i biased_exp = _mm256_add_epi64(M, bias);
    const __m256i one = _mm256_set1_epi64x(1);
    const __m256i normal_shift = _mm256_set1_epi64x(EXPONENT_SHIFT);
    const __m256i shift = min_epi64(
        normal_shift,
        _mm256_sub_epi64(_mm256_add_epi64(normal_shift, biased_exp), one)
    );
    const __m256i shift_mask = _mm256_cmpeq_epi64(shift, normal_shift);
    __m256i shifted_value = _mm256_blendv_epi8(one, biased_exp, shift_mask);
    __m256i mul_mask = _mm256_cmpeq_epi64(M, _mm256_set1_epi64x(BIAS + 1));
    mul_mask = _mm256_and_si256(mul_mask, one);
    shifted_value = _mm256_sub_epi64(shifted_value, mul_mask);
    const __m256i large_exponent_mask =
        _mm256_cmpgt_epi64(M, _mm256_set1_epi64x(BIAS + 1));
    __m256d two_M = _mm256_castsi256_pd(_mm256_sllv_epi64(shifted_value, shift));
    two_M = _mm256_blendv_pd(
        two_M,
        _mm256_set1_pd(std::numeric_limits<double>::infinity()),
        _mm256_castsi256_pd(large_exponent_mask)
    );

    const __m256d p2 = _mm256_add_pd(S_lead_vec, _mm256_fmadd_pd(S, P, S_trail_vec));

    mul_mask = _mm256_add_epi64(mul_mask, bias);
    __m256d res = _mm256_mul_pd(
        _mm256_mul_pd(two_M, p2),
        _mm256_castsi256_pd(_mm256_slli_epi64(mul_mask, EXPONENT_SHIFT))
    );

    // special cases
    const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    const __m256d minus_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    const __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    const __m256d abs_x_lt_t2_mask =
        _mm256_cmp_pd(abs_x, _mm256_set1_pd(THRESHOLD_2), _CMP_LT_OQ);

    const __m256d x_over_t1_mask =
        _mm256_cmp_pd(x, _mm256_set1_pd(THRESHOLD_1), _CMP_GT_OQ);
    const __m256d x_neg_over_t1_mask =
        _mm256_cmp_pd(x, _mm256_set1_pd(-THRESHOLD_1), _CMP_LT_OQ);

    const __m256d inf_mask = _mm256_cmp_pd(x, inf, _CMP_EQ_OQ);
    const __m256d minus_inf_mask = _mm256_cmp_pd(x, minus_inf, _CMP_EQ_OQ);
    const __m256d input_nan_mask = _mm256_cmp_pd(x, x, _CMP_EQ_OQ);

    res = _mm256_blendv_pd(res, inf, _mm256_or_pd(inf_mask, x_over_t1_mask));
    res = _mm256_blendv_pd(
        res,
        _mm256_set1_pd(0.0),
        _mm256_or_pd(minus_inf_mask, x_neg_over_t1_mask)
    );
    res = _mm256_blendv_pd(res, _mm256_add_pd(_mm256_set1_pd(1.0), x), abs_x_lt_t2_mask);
    res = _mm256_blendv_pd(x, res, input_nan_mask);

    return res;
}

inline __m256 exp8f(__m256 x) {
    using namespace constants::simple_precision;

    const __m256 N = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(INV_L)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    const __m256i Ni = _mm256_cvtps_epi32(N);
    const __m256i N2i =
        _mm256_and_si256(Ni, _mm256_set1_epi32(31));  // N2i = Ni % 32, N2i >= 0
    const __m256i N1i = Ni - N2i;
    const __m256 N2 = _mm256_cvtepi32_ps(N2i);
    const __m256 N1 = _mm256_cvtepi32_ps(N1i);

    const __m256 abs_N = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), N);
    const __m256 n_ge_512 = _mm256_cmp_ps(abs_N, _mm256_set1_ps(512), _CMP_GE_OQ);

    const __m256 vec_L1 = _mm256_set1_ps(L1);
    const __m256 R1_ge_512 =
        _mm256_fnmadd_ps(N2, vec_L1, _mm256_fnmadd_ps(N1, vec_L1, x));
    const __m256 R1_lt_512 = _mm256_fnmadd_ps(N, vec_L1, x);
    const __m256 R1 = _mm256_blendv_ps(R1_lt_512, R1_ge_512, n_ge_512);
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
    const __m256i shift_mask = _mm256_cmpeq_epi32(shift, normal_shift);
    __m256i shifted_value = _mm256_blendv_epi8(one, biased_exp, shift_mask);
    __m256i mul_mask = _mm256_cmpeq_epi32(M, _mm256_set1_epi32(BIAS + 1));
    mul_mask = _mm256_and_si256(mul_mask, one);
    shifted_value = _mm256_sub_epi32(shifted_value, mul_mask);
    const __m256i large_exponent_mask =
        _mm256_cmpgt_epi32(M, _mm256_set1_epi32(BIAS + 1));
    __m256 two_M = _mm256_castsi256_ps(_mm256_sllv_epi32(shifted_value, shift));
    two_M = _mm256_blendv_ps(
        two_M,
        _mm256_set1_ps(std::numeric_limits<float>::infinity()),
        _mm256_castsi256_ps(large_exponent_mask)
    );

    const __m256 p2 = _mm256_add_ps(S_lead_vec, _mm256_fmadd_ps(S, P, S_trail_vec));

    mul_mask = _mm256_add_epi32(mul_mask, bias);
    __m256 res = _mm256_mul_ps(
        _mm256_mul_ps(two_M, p2),
        _mm256_castsi256_ps(_mm256_slli_epi32(mul_mask, EXPONENT_SHIFT))
    );

    // special cases
    const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    const __m256 minus_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

    const __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    const __m256 abs_x_lt_t2_mask =
        _mm256_cmp_ps(abs_x, _mm256_set1_ps(THRESHOLD_2), _CMP_LT_OQ);

    const __m256 x_over_t1_mask =
        _mm256_cmp_ps(x, _mm256_set1_ps(THRESHOLD_1), _CMP_GT_OQ);
    const __m256 x_neg_over_t1_mask =
        _mm256_cmp_ps(x, _mm256_set1_ps(-THRESHOLD_1), _CMP_LT_OQ);

    const __m256 inf_mask = _mm256_cmp_ps(x, inf, _CMP_EQ_OQ);
    const __m256 minus_inf_mask = _mm256_cmp_ps(x, minus_inf, _CMP_EQ_OQ);
    const __m256 input_nan_mask = _mm256_cmp_ps(x, x, _CMP_EQ_OQ);

    res = _mm256_blendv_ps(res, inf, _mm256_or_ps(inf_mask, x_over_t1_mask));
    res = _mm256_blendv_ps(
        res,
        _mm256_set1_ps(0.0f),
        _mm256_or_ps(minus_inf_mask, x_neg_over_t1_mask)
    );
    res =
        _mm256_blendv_ps(res, _mm256_add_ps(_mm256_set1_ps(1.0f), x), abs_x_lt_t2_mask);
    res = _mm256_blendv_ps(x, res, input_nan_mask);

    return res;
}

inline __m128 exp4f(__m128 x) {
    using namespace constants::simple_precision;

    const __m128 N = _mm_round_ps(
        _mm_mul_ps(x, _mm_set1_ps(INV_L)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    const __m128i Ni = _mm_cvtps_epi32(N);
    const __m128i N2i =
        _mm_and_si128(Ni, _mm_set1_epi32(31));  // N2i = Ni % 32, N2i >= 0
    const __m128i N1i = Ni - N2i;
    const __m128 N2 = _mm_cvtepi32_ps(N2i);
    const __m128 N1 = _mm_cvtepi32_ps(N1i);

    const __m128 abs_N = _mm_andnot_ps(_mm_set1_ps(-0.0f), N);
    const __m128 n_ge_512 = _mm_cmp_ps(abs_N, _mm_set1_ps(512), _CMP_GE_OQ);

    const __m128 vec_L1 = _mm_set1_ps(L1);
    const __m128 R1_ge_512 = _mm_fnmadd_ps(N2, vec_L1, _mm_fnmadd_ps(N1, vec_L1, x));
    const __m128 R1_lt_512 = _mm_fnmadd_ps(N, vec_L1, x);
    const __m128 R1 = _mm_blendv_ps(R1_lt_512, R1_ge_512, n_ge_512);
    const __m128 R2 = _mm_fnmadd_ps(N, _mm_set1_ps(L2), _mm_set1_ps(0.0f));

    const __m128i M = _mm_srai_epi32(N1i, 5);  // M = N1i / 32

    const __m128 R = _mm_add_ps(R1, R2);
    __m128 Q = _mm_fmadd_ps(R, _mm_set1_ps(A2), _mm_set1_ps(A1));
    Q = _mm_mul_ps(R, Q);
    __m128 P = _mm_add_ps(R1, _mm_fmadd_ps(R, Q, R2));

    const __m128 S_lead_vec = _mm_i32gather_ps(S_lead.data(), N2i, 4);
    const __m128 S_trail_vec = _mm_i32gather_ps(S_trail.data(), N2i, 4);

    const __m128 S = _mm_add_ps(S_lead_vec, S_trail_vec);

    // compute 2^M
    const __m128i bias = _mm_set1_epi32(BIAS);
    const __m128i biased_exp = _mm_add_epi32(M, bias);
    const __m128i one = _mm_set1_epi32(1);
    const __m128i normal_shift = _mm_set1_epi32(EXPONENT_SHIFT);
    const __m128i shift = _mm_min_epi32(
        normal_shift,
        _mm_sub_epi32(_mm_add_epi32(normal_shift, biased_exp), one)
    );
    const __m128i shift_mask = _mm_cmpeq_epi32(shift, normal_shift);
    __m128i shifted_value = _mm_blendv_epi8(one, biased_exp, shift_mask);
    __m128i mul_mask = _mm_cmpeq_epi32(M, _mm_set1_epi32(BIAS + 1));
    mul_mask = _mm_and_si128(mul_mask, one);
    shifted_value = _mm_sub_epi32(shifted_value, mul_mask);
    const __m128i large_exponent_mask = _mm_cmpgt_epi32(M, _mm_set1_epi32(BIAS + 1));
    __m128 two_M = _mm_castsi128_ps(_mm_sllv_epi32(shifted_value, shift));
    two_M = _mm_blendv_ps(
        two_M,
        _mm_set1_ps(std::numeric_limits<float>::infinity()),
        _mm_castsi128_ps(large_exponent_mask)
    );

    const __m128 p2 = _mm_add_ps(S_lead_vec, _mm_fmadd_ps(S, P, S_trail_vec));

    mul_mask = _mm_add_epi32(mul_mask, bias);
    __m128 res = _mm_mul_ps(
        _mm_mul_ps(two_M, p2),
        _mm_castsi128_ps(_mm_slli_epi32(mul_mask, EXPONENT_SHIFT))
    );

    // special cases
    const __m128 inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
    const __m128 minus_inf = _mm_set1_ps(-std::numeric_limits<float>::infinity());

    const __m128 abs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
    const __m128 abs_x_lt_t2_mask =
        _mm_cmp_ps(abs_x, _mm_set1_ps(THRESHOLD_2), _CMP_LT_OQ);

    const __m128 x_over_t1_mask = _mm_cmp_ps(x, _mm_set1_ps(THRESHOLD_1), _CMP_GT_OQ);
    const __m128 x_neg_over_t1_mask =
        _mm_cmp_ps(x, _mm_set1_ps(-THRESHOLD_1), _CMP_LT_OQ);

    const __m128 inf_mask = _mm_cmp_ps(x, inf, _CMP_EQ_OQ);
    const __m128 minus_inf_mask = _mm_cmp_ps(x, minus_inf, _CMP_EQ_OQ);
    const __m128 input_nan_mask = _mm_cmp_ps(x, x, _CMP_EQ_OQ);

    res = _mm_blendv_ps(res, inf, _mm_or_ps(inf_mask, x_over_t1_mask));
    res = _mm_blendv_ps(
        res,
        _mm_set1_ps(0.0f),
        _mm_or_ps(minus_inf_mask, x_neg_over_t1_mask)
    );
    res = _mm_blendv_ps(res, _mm_add_ps(_mm_set1_ps(1.0f), x), abs_x_lt_t2_mask);
    res = _mm_blendv_ps(x, res, input_nan_mask);

    return res;
}
}  // namespace avx2

#endif
