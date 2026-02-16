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
// constexpr const tuple& operator=(tuple<UTypes...>&& u) const;
//
// Constraints:
// - sizeof...(Types) equals sizeof...(UTypes) and
// - (is_assignable_v<const Types&, UTypes> && ...) is true.

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <type_traits>

#include <tuple.hpp>
#if defined(CEXA_HAS_CXX23)
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/copy_move_types.h>

// test constraints

// sizeof...(Types) equals sizeof...(UTypes)
static_assert(std::is_assignable_v<const cexa::tuple<int&>&, cexa::tuple<long&>&&>);
static_assert(!std::is_assignable_v<const cexa::tuple<int&, int&>&, cexa::tuple<long&>&&>);
static_assert(!std::is_assignable_v<const cexa::tuple<int&>&, cexa::tuple<long&, long&>&&>);

// (is_assignable_v<const Types&, UTypes&&> && ...) is true
static_assert(std::is_assignable_v<const cexa::tuple<AssignableFrom<ConstMoveAssign>>&, cexa::tuple<ConstMoveAssign>&&>);

static_assert(std::is_assignable_v<const cexa::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign>&,
                                   cexa::tuple<ConstMoveAssign, ConstMoveAssign>&&>);

static_assert(!std::is_assignable_v<const cexa::tuple<AssignableFrom<ConstMoveAssign>, AssignableFrom<MoveAssign>>&,
                                    cexa::tuple<ConstMoveAssign, MoveAssign>&&>);

KOKKOS_INLINE_FUNCTION constexpr bool test() {
  // reference types
  {
    int i1 = 1;
    int i2 = 2;
    long j1 = 3;
    long j2 = 4;
    cexa::tuple<int&, int&> t1{i1, i2};
    const cexa::tuple<long&, long&> t2{j1, j2};
    t2 = std::move(t1);
    CEXA_EXPECT_EQ(cexa::get<0>(t2), 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t2), 2);
  }

  // user defined const copy assignment
  {
    cexa::tuple<ConstMoveAssign> t1{1};
    const cexa::tuple<AssignableFrom<ConstMoveAssign>> t2{2};
    t2 = std::move(t1);
    CEXA_EXPECT_EQ(cexa::get<0>(t2).v.val, 1);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    cexa::tuple<TracedAssignment> t1{};
    const cexa::tuple<AssignableFrom<TracedAssignment>> t2{};
    t2 = std::move(t1);
    CEXA_EXPECT_EQ(cexa::get<0>(t2).v.constMoveAssign, 1);
  }

  return true;
}

// clang-format off
CEXA_TEST(tuple_assign, const_convert_move, (
  test();

  // FIXME: gcc cannot have mutable member in constant expression
#if !defined(KOKKOS_COMPILER_GNU) || KOKKOS_COMPILER_GNU >= 1300
  static_assert(test());
#endif
))
// clang-format on
#endif
