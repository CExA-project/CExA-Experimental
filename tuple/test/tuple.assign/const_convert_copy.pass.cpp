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

// template <class... UTypes>
// constexpr const tuple& operator=(const tuple<UTypes...>& u) const;
//
// Constraints:
// - sizeof...(Types) equals sizeof...(UTypes) and
// - (is_assignable_v<const Types&, const UTypes&> && ...) is true.

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <type_traits>

#include <tuple.hpp>
#if defined(CEXA_HAS_CXX23)
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/copy_move_types.h>

// test constraints

// sizeof...(Types) equals sizeof...(UTypes)
static_assert(std::is_assignable_v<const cexa::tuple<int&>&, const cexa::tuple<long&>&>);
static_assert(!std::is_assignable_v<const cexa::tuple<int&, int&>&, const cexa::tuple<long&>&>);
static_assert(!std::is_assignable_v<const cexa::tuple<int&>&, const cexa::tuple<long&, long&>&>);

// (is_assignable_v<const Types&, const UTypes&> && ...) is true
static_assert(std::is_assignable_v<const cexa::tuple<AssignableFrom<ConstCopyAssign>>&,
                                   const cexa::tuple<ConstCopyAssign>&>);

static_assert(std::is_assignable_v<const cexa::tuple<AssignableFrom<ConstCopyAssign>, ConstCopyAssign>&,
                                   const cexa::tuple<ConstCopyAssign, ConstCopyAssign>&>);

static_assert(!std::is_assignable_v<const cexa::tuple<AssignableFrom<ConstCopyAssign>, CopyAssign>&,
                                    const cexa::tuple<ConstCopyAssign, CopyAssign>&>);

KOKKOS_INLINE_FUNCTION constexpr bool test() {
  // reference types
  {
    int i1 = 1;
    int i2 = 2;
    long j1 = 3;
    long j2 = 4;
    const cexa::tuple<int&, int&> t1{i1, i2};
    const cexa::tuple<long&, long&> t2{j1, j2};
    t2 = t1;
    // CEXA_EXPECT_EQ(cexa::get<0>(t2), 1);
    // CEXA_EXPECT_EQ(cexa::get<1>(t2), 2);
  }

  // user defined const copy assignment
  {
    const cexa::tuple<ConstCopyAssign> t1{1};
    const cexa::tuple<AssignableFrom<ConstCopyAssign>> t2{2};
    t2 = t1;
    CEXA_EXPECT_EQ(cexa::get<0>(t2).v.val, 1);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    cexa::tuple<TracedAssignment> t1{};
    const cexa::tuple<AssignableFrom<TracedAssignment>> t2{};
    t2 = t1;
    CEXA_EXPECT_EQ(cexa::get<0>(t2).v.constCopyAssign, 1);
  }

  return true;
}

// clang-format off
CEXA_TEST(tuple_assign, const_convert_copy, (
  test();

  // FIXME: gcc cannot have mutable member in constant expression
#if !defined(KOKKOS_COMPILER_GNU) || KOKKOS_COMPILER_GNU >= 1300
  static_assert(test());
#endif
))
// clang-format on
#endif
