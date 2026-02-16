// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
//
// This is a modified version of the tuple tests from llvm's libcxx tests,
// below is the original copyright statement
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// void swap(const tuple& rhs);

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <utility>

#include <tuple.hpp>
#if defined(CEXA_HAS_CXX23)
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

#ifndef TEST_HAS_NO_EXCEPTIONS
class SwapThrower {
  void swap(SwapThrower&) = delete;
  void swap(const SwapThrower&) const = delete;
};

void swap(const SwapThrower&, const SwapThrower&) { throw 0.f; }

static_assert(std::is_swappable_v<const SwapThrower>);
static_assert(std::is_swappable_with_v<const SwapThrower&, const SwapThrower&>);

void test_noexcept() {
  using std::swap;

  const cexa::tuple<SwapThrower> t1;
  const cexa::tuple<SwapThrower> t2;

  try {
    t1.swap(t2);
    swap(t1, t2);
    CEXA_EXPECT(false);
  } catch (float) {
  }

  try {
    swap(std::as_const(t1), std::as_const(t2));
    CEXA_EXPECT(false);
  } catch (float) {
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

struct ConstSwappable {
  mutable int i;
};

KOKKOS_INLINE_FUNCTION
#if !defined(KOKKOS_COMPILER_GNU) || TEST_STD_VER > 17
constexpr
#endif
void swap(const ConstSwappable& lhs, const ConstSwappable& rhs) {
  // FIXME: a std::swap here doesn't swap the values on cuda 12.2 + gcc 11.2
#if defined(KOKKOS_COMPILER_NVCC) && defined(KOKKOS_COMPILER_GNU) && KOKKOS_COMPILER_GNU < 1200
  auto tmp = lhs.i;
  lhs.i = rhs.i;
  rhs.i = tmp;
#else
  std::swap(lhs.i, rhs.i);
#endif
}

KOKKOS_INLINE_FUNCTION constexpr bool test() {
  {
    typedef cexa::tuple<const ConstSwappable> T;
    const T t0(ConstSwappable{0});
    T t1(ConstSwappable{1});
    t0.swap(t1);
    CEXA_EXPECT_EQ(cexa::get<0>(t0).i, 1);
    CEXA_EXPECT_EQ(cexa::get<0>(t1).i, 0);
  }
  {
    typedef cexa::tuple<ConstSwappable, ConstSwappable> T;
    const T t0({0}, {1});
    const T t1({2}, {3});
    t0.swap(t1);
    CEXA_EXPECT_EQ(cexa::get<0>(t0).i, 2);
    CEXA_EXPECT_EQ(cexa::get<1>(t0).i, 3);
    CEXA_EXPECT_EQ(cexa::get<0>(t1).i, 0);
    CEXA_EXPECT_EQ(cexa::get<1>(t1).i, 1);
  }
  {
    typedef cexa::tuple<ConstSwappable, const ConstSwappable, const ConstSwappable> T;
    const T t0({0}, {1}, {2});
    const T t1({3}, {4}, {5});
    t0.swap(t1);
    CEXA_EXPECT_EQ(cexa::get<0>(t0).i, 3);
    CEXA_EXPECT_EQ(cexa::get<1>(t0).i, 4);
    CEXA_EXPECT_EQ(cexa::get<2>(t0).i, 5);
    CEXA_EXPECT_EQ(cexa::get<0>(t1).i, 0);
    CEXA_EXPECT_EQ(cexa::get<1>(t1).i, 1);
    CEXA_EXPECT_EQ(cexa::get<2>(t1).i, 2);
  }
  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
TEST(host_tuple_swap, member_swap_const_host) {
  test_noexcept();
}
#endif

// clang-format off
CEXA_TEST(tuple_swap, member_swap_const, (
  test();

// FIXME: gcc cannot have mutable member in constant expression
// FIXME: see why this is not a constant expresion in c++17
#if TEST_STD_VER > 17 && (!defined(KOKKOS_COMPILER_GNU) || KOKKOS_COMPILER_GNU >= 1300)
  static_assert(test());
#endif
))
// clang-format on
#endif
