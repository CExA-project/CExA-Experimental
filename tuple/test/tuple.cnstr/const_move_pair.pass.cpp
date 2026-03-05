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
// constexpr explicit(see below) tuple<Types...>::tuple(const pair<U1, U2>&& u);

// Constraints:
// - sizeof...(Types) is 2 and
// - is_constructible_v<T0, decltype(get<0>(FWD(u)))> is true and
// - is_constructible_v<T1, decltype(get<1>(FWD(u)))> is true.

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <utility>

#include <tuple.hpp>
#if defined(CEXA_HAS_CXX23)

#include <support/cexa_test_macros.hpp>
#include <support/copy_move_types.h>

// test constraints
// sizeof...(Types) == 2
static_assert(std::is_constructible_v<cexa::tuple<ConstMove, int>, const std::pair<ConstMove, int>&&>);

static_assert(!std::is_constructible_v<cexa::tuple<ConstMove>, const std::pair<ConstMove, int>&&>);

static_assert(!std::is_constructible_v<cexa::tuple<ConstMove, int, int>, const std::pair<ConstMove, int>&&>);

// test constraints
// is_constructible_v<T0, decltype(get<0>(FWD(u)))> is true and
// is_constructible_v<T1, decltype(get<1>(FWD(u)))> is true.
static_assert(std::is_constructible_v<cexa::tuple<int, int>, const std::pair<int, int>&&>);

static_assert(!std::is_constructible_v<cexa::tuple<NoConstructorFromInt, int>, const std::pair<int, int>&&>);

static_assert(!std::is_constructible_v<cexa::tuple<int, NoConstructorFromInt>, const std::pair<int, int>&&>);

static_assert(
    !std::is_constructible_v< cexa::tuple<NoConstructorFromInt, NoConstructorFromInt>, const std::pair<int, int>&&>);

// test: The expression inside explicit is equivalent to:
// !is_convertible_v<decltype(get<0>(FWD(u))), T0> ||
// !is_convertible_v<decltype(get<1>(FWD(u))), T1>
static_assert(std::is_convertible_v<const std::pair<ConstMove, ConstMove>&&,
                                    cexa::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>>>);

static_assert(!std::is_convertible_v<const std::pair<ConstMove, ConstMove>&&,
                                     cexa::tuple<ExplicitConstructibleFrom<ConstMove>, ConvertibleFrom<ConstMove>>>);

static_assert(!std::is_convertible_v<const std::pair<ConstMove, ConstMove>&&,
                                     cexa::tuple<ConvertibleFrom<ConstMove>, ExplicitConstructibleFrom<ConstMove>>>);
// clang-format off
// TODO: see why we cannot use Kokkos::pair here
TEST(tuple_cnstr, const_move_pair) {
  // test implicit conversions.
  {
    const std::pair<ConstMove, int> p{1, 2};
    cexa::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<int>> t = std::move(p);
    CEXA_EXPECT_EQ(cexa::get<0>(t).v.val, 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t).v, 2);
  }

  // test explicit conversions.
  {
    const std::pair<ConstMove, int> p{1, 2};
    cexa::tuple<ExplicitConstructibleFrom<ConstMove>, ExplicitConstructibleFrom<int>> t{std::move(p)};
    CEXA_EXPECT_EQ(cexa::get<0>(t).v.val, 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t).v, 2);
  }

  // non const overload should be called
  {
    const std::pair<TracedCopyMove, TracedCopyMove> p;
    cexa::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = std::move(p);
    CEXA_EXPECT(constMoveCtrCalled(cexa::get<0>(t).v));
    CEXA_EXPECT(constMoveCtrCalled(cexa::get<1>(t)));
  }
}
// clang-format on
#endif
