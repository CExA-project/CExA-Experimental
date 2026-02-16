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

// template <class... Types> class tuple;

// template <class... Types>
//   void swap(const tuple<Types...>& x, const tuple<Types...>& y);

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <tuple.hpp>
#if defined(CEXA_HAS_CXX23)
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct S {
  int* calls;
  KOKKOS_INLINE_FUNCTION friend constexpr void swap(S& a, S& b) {
    *a.calls += 1;
    *b.calls += 1;
  }
};
struct CS {
  int* calls;
  KOKKOS_INLINE_FUNCTION friend constexpr void swap(const CS& a, const CS& b) {
    *a.calls += 1;
    *b.calls += 1;
  }
};

static_assert(std::is_swappable_v<cexa::tuple<>>);
static_assert(std::is_swappable_v<cexa::tuple<S>>);
static_assert(std::is_swappable_v<cexa::tuple<CS>>);
static_assert(std::is_swappable_v<cexa::tuple<S&>>);
static_assert(std::is_swappable_v<cexa::tuple<CS, S>>);
static_assert(std::is_swappable_v<cexa::tuple<CS, S&>>);
static_assert(std::is_swappable_v<const cexa::tuple<>>);
static_assert(!std::is_swappable_v<const cexa::tuple<S>>);
static_assert(std::is_swappable_v<const cexa::tuple<CS>>);
static_assert(std::is_swappable_v<const cexa::tuple<S&>>);
static_assert(!std::is_swappable_v<const cexa::tuple<CS, S>>);
static_assert(std::is_swappable_v<const cexa::tuple<CS, S&>>);

KOKKOS_INLINE_FUNCTION constexpr bool test() {
  int cs_calls = 0;
  int s_calls = 0;
  S s1{&s_calls};
  S s2{&s_calls};
  const cexa::tuple<CS, S&> t1 = {CS{&cs_calls}, s1};
  const cexa::tuple<CS, S&> t2 = {CS{&cs_calls}, s2};
  swap(t1, t2);
  CEXA_EXPECT_EQ(cs_calls, 2);
  CEXA_EXPECT_EQ(s_calls, 2);

  return true;
}

// clang-format off
CEXA_TEST(tuple_special, non_member_swap_const, (
  test();
  static_assert(test());
))
// clang-format on
#endif
