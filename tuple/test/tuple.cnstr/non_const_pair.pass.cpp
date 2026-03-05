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

// template <class... Types>
// template <class U1, class U2>
// constexpr explicit(see below) tuple<Types...>::tuple(pair<U1, U2>& u);

// Constraints:
// - sizeof...(Types) is 2 and
// - is_constructible_v<T0, decltype(get<0>(FWD(u)))> is true and
// - is_constructible_v<T1, decltype(get<1>(FWD(u)))> is true.

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <utility>

#include <tuple.hpp>
#if defined(CEXA_HAS_CXX23)

#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/copy_move_types.h>

// test constraints
// sizeof...(Types) == 2
static_assert(std::is_constructible_v<cexa::tuple<MutableCopy, int>, std::pair<MutableCopy, int>&>);

static_assert(!std::is_constructible_v<cexa::tuple<MutableCopy>, std::pair<MutableCopy, int>&>);

static_assert(!std::is_constructible_v<cexa::tuple<MutableCopy, int, int>, std::pair<MutableCopy, int>&>);

// test constraints
// is_constructible_v<T0, decltype(get<0>(FWD(u)))> is true and
// is_constructible_v<T1, decltype(get<1>(FWD(u)))> is true.
static_assert(std::is_constructible_v<cexa::tuple<int, int>, std::pair<int, int>&>);

static_assert(!std::is_constructible_v<cexa::tuple<NoConstructorFromInt, int>, std::pair<int, int>&>);

static_assert(!std::is_constructible_v<cexa::tuple<int, NoConstructorFromInt>, std::pair<int, int>&>);

static_assert(!std::is_constructible_v< cexa::tuple<NoConstructorFromInt, NoConstructorFromInt>, std::pair<int, int>&>);

// test: The expression inside explicit is equivalent to:
// !is_convertible_v<decltype(get<0>(FWD(u))), T0> ||
// !is_convertible_v<decltype(get<1>(FWD(u))), T1>
static_assert(std::is_convertible_v<std::pair<MutableCopy, MutableCopy>&,
                                    cexa::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>>);

static_assert(!std::is_convertible_v<std::pair<MutableCopy, MutableCopy>&,
                                     cexa::tuple<ExplicitConstructibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>>);

static_assert(!std::is_convertible_v<std::pair<MutableCopy, MutableCopy>&,
                                     cexa::tuple<ConvertibleFrom<MutableCopy>, ExplicitConstructibleFrom<MutableCopy>>>);
// TODO: Add a Kokkos::pair test
constexpr bool test() {
  // test implicit conversions.
  {
    std::pair<MutableCopy, int> p{1, 2};
    cexa::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<int>> t = p;
    CEXA_EXPECT_EQ(cexa::get<0>(t).v.val, 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t).v, 2);
  }

  // test explicit conversions.
  {
    std::pair<MutableCopy, int> p{1, 2};
    cexa::tuple<ExplicitConstructibleFrom<MutableCopy>, ExplicitConstructibleFrom<int>> t{p};
    CEXA_EXPECT_EQ(cexa::get<0>(t).v.val, 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t).v, 2);
  }

  // non const overload should be called
  {
    std::pair<TracedCopyMove, TracedCopyMove> p;
    cexa::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = p;
    CEXA_EXPECT(nonConstCopyCtrCalled(cexa::get<0>(t).v));
    CEXA_EXPECT(nonConstCopyCtrCalled(cexa::get<1>(t)));
  }

  return true;
}

TEST(host_tuple_cnstr, non_const_pair) {
  test();
  static_assert(test());
}
#endif
