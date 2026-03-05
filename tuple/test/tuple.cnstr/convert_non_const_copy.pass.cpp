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
// template <class... UTypes>
//   constexpr explicit(see below) tuple<Types>::tuple(tuple<UTypes...>&);
//
// Constraints:
//  sizeof...(Types) equals sizeof...(UTypes) &&
//  (is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...) is true &&
//  (
//    sizeof...(Types) is not 1 ||
//    (
//      !is_convertible_v<decltype(u), T> &&
//      !is_constructible_v<T, decltype(u)> &&
//      !is_same_v<T, U>
//    )
//  )

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20


#include <tuple.hpp>
#if defined(CEXA_HAS_CXX23)

#include <support/cexa_test_macros.hpp>
#include <support/copy_move_types.h>

// test: The expression inside explicit is equivalent to:
// !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
static_assert(std::is_convertible_v<cexa::tuple<MutableCopy>&, cexa::tuple<ConvertibleFrom<MutableCopy>>>);

static_assert(std::is_convertible_v<cexa::tuple<MutableCopy, MutableCopy>&,
                                    cexa::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>>);

static_assert(!std::is_convertible_v<cexa::tuple<MutableCopy>&, cexa::tuple<ExplicitConstructibleFrom<MutableCopy>>>);

static_assert(!std::is_convertible_v<cexa::tuple<MutableCopy, MutableCopy>&,
                                     cexa::tuple<ConvertibleFrom<MutableCopy>, ExplicitConstructibleFrom<MutableCopy>>>);
// test constraints

// sizeof...(Types) != sizeof...(UTypes)
static_assert(!std::is_constructible_v<cexa::tuple<int, int>, cexa::tuple<int>&>);
static_assert(!std::is_constructible_v<cexa::tuple<int>, cexa::tuple<int, int>&>);
static_assert(!std::is_constructible_v<cexa::tuple<int, int, int>, cexa::tuple<int, int>&>);

// !(is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...)
static_assert(!std::is_constructible_v<cexa::tuple<int, NoConstructorFromInt>, cexa::tuple<int, int>&>);

// clang-format off
CEXA_TEST(tuple_cnstr, convert_non_const_copy, (
  // test implicit conversions.
  // sizeof...(Types) == 1
  {
    cexa::tuple<MutableCopy> t1{1};
    cexa::tuple<ConvertibleFrom<MutableCopy>> t2 = t1;
    CEXA_EXPECT_EQ(cexa::get<0>(t2).v.val, 1);
  }

  // test implicit conversions.
  // sizeof...(Types) > 1
  {
    cexa::tuple<MutableCopy, int> t1{1, 2};
    cexa::tuple<ConvertibleFrom<MutableCopy>, int> t2 = t1;
    CEXA_EXPECT_EQ(cexa::get<0>(t2).v.val, 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t2), 2);
  }

  // test explicit conversions.
  // sizeof...(Types) == 1
  {
    cexa::tuple<MutableCopy> t1{1};
    cexa::tuple<ExplicitConstructibleFrom<MutableCopy>> t2{t1};
    CEXA_EXPECT_EQ(cexa::get<0>(t2).v.val, 1);
  }

  // test explicit conversions.
  // sizeof...(Types) > 1
  {
    cexa::tuple<MutableCopy, int> t1{1, 2};
    cexa::tuple<ExplicitConstructibleFrom<MutableCopy>, int> t2{t1};
    CEXA_EXPECT_EQ(cexa::get<0>(t2).v.val, 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t2), 2);
  }

  // test constraints

  // sizeof...(Types) == 1 && other branch of "||" satisfied
  {
    cexa::tuple<TracedCopyMove> t1{};
    cexa::tuple<ConvertibleFrom<TracedCopyMove>> t2{t1};
    CEXA_EXPECT(nonConstCopyCtrCalled(cexa::get<0>(t2).v));
  }

  // sizeof...(Types) == 1 && is_same_v<T, U>
  {
    cexa::tuple<TracedCopyMove> t1{};
    cexa::tuple<TracedCopyMove> t2{t1};
    CEXA_EXPECT(!nonConstCopyCtrCalled(cexa::get<0>(t2)));
  }

  // sizeof...(Types) != 1
  {
    cexa::tuple<TracedCopyMove, TracedCopyMove> t1{};
    cexa::tuple<TracedCopyMove, TracedCopyMove> t2{t1};
    CEXA_EXPECT(nonConstCopyCtrCalled(cexa::get<0>(t2)));
  }

  // NOTE: original comment: These two test points cause gcc to ICE
// #if !defined(TEST_COMPILER_GCC)
  // sizeof...(Types) == 1 && is_convertible_v<decltype(u), T>
#if !defined(KOKKOS_COMPILER_GNU) || KOKKOS_COMPILER_GNU >= 1300
  {
    cexa::tuple<CvtFromTupleRef> t1{};
    cexa::tuple<ConvertibleFrom<CvtFromTupleRef>> t2{t1};
    CEXA_EXPECT(!nonConstCopyCtrCalled(cexa::get<0>(t2).v));
  }

  // sizeof...(Types) == 1 && is_constructible_v<decltype(u), T>
  {
    cexa::tuple<ExplicitCtrFromTupleRef> t1{};
    cexa::tuple<ConvertibleFrom<ExplicitCtrFromTupleRef>> t2{t1};
    CEXA_EXPECT(!nonConstCopyCtrCalled(cexa::get<0>(t2).v));
  }
#endif
))
// clang-format off
#endif
