// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <CEXA_SIMD_Backends.hpp>

using simd_type          = Kokkos::Experimental::simd<float>;
constexpr int simd_width = simd_type::size();

#define TEST_UNARY_FUNC(FUNC)                                           \
  TEST(unary_functions, FUNC) {                                         \
    float values[simd_width];                                           \
                                                                        \
    const float inf = std::numeric_limits<float>::infinity();           \
                                                                        \
    values[0] = -inf;                                                   \
    for (int i = 1; i < simd_width; i++) {                              \
      values[i] = std::nextafter(values[i - 1], inf);                   \
    }                                                                   \
                                                                        \
    while (values[simd_width - 1] < inf) {                              \
      simd_type x(values, Kokkos::Experimental::simd_flag_default);     \
      simd_type res = Kokkos::FUNC(x);                                  \
                                                                        \
      for (int i = 0; i < simd_width; i++) {                            \
        float expected = std::FUNC(values[i]);                          \
        EXPECT_FLOAT_EQ(res[i], expected) << "For value " << values[i]; \
      }                                                                 \
                                                                        \
      values[0] = std::nextafter(values[simd_width - 1], inf);          \
      for (int i = 1; i < simd_width; i++) {                            \
        values[i] = std::nextafter(values[i - 1], inf);                 \
      }                                                                 \
    }                                                                   \
  }

TEST_UNARY_FUNC(exp)
TEST_UNARY_FUNC(exp2)
TEST_UNARY_FUNC(log)
TEST_UNARY_FUNC(log10)
TEST_UNARY_FUNC(log2)
TEST_UNARY_FUNC(cbrt)
TEST_UNARY_FUNC(sin)
TEST_UNARY_FUNC(cos)
TEST_UNARY_FUNC(tan)
TEST_UNARY_FUNC(asin)
TEST_UNARY_FUNC(acos)
TEST_UNARY_FUNC(atan)
TEST_UNARY_FUNC(sinh)
TEST_UNARY_FUNC(cosh)
TEST_UNARY_FUNC(tanh)
TEST_UNARY_FUNC(asinh)
TEST_UNARY_FUNC(acosh)
TEST_UNARY_FUNC(atanh)
TEST_UNARY_FUNC(erf)
TEST_UNARY_FUNC(erfc)
#if defined(CEXA_ENABLE_SLEEF)
TEST_UNARY_FUNC(tgamma)
TEST_UNARY_FUNC(lgamma)
#endif

// TODO: test binary and ternary functions

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  return result;
}
