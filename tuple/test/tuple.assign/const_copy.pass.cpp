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

// constexpr const tuple& operator=(const tuple&) const;
//
// Constraints: (is_copy_assignable_v<const Types> && ...) is true.

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// test constraints

#include <type_traits>

#include <tuple.hpp>
#if defined(CEXA_HAS_CXX23)
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/copy_move_types.h>

static_assert(!std::is_assignable_v<const cexa::tuple<int>&, const cexa::tuple<int>&>);
static_assert(std::is_assignable_v<const cexa::tuple<int&>&, const cexa::tuple<int&>&>);
static_assert(std::is_assignable_v<const cexa::tuple<int&, int&>&, const cexa::tuple<int&, int&>&>);
static_assert(!std::is_assignable_v<const cexa::tuple<int&, int>&, const cexa::tuple<int&, int>&>);
static_assert(std::is_assignable_v<const cexa::tuple<ConstCopyAssign>&, const cexa::tuple<ConstCopyAssign>&>);
static_assert(!std::is_assignable_v<const cexa::tuple<CopyAssign>&, const cexa::tuple<CopyAssign>&>);
static_assert(!std::is_assignable_v<const cexa::tuple<ConstMoveAssign>&, const cexa::tuple<ConstMoveAssign>&>);
static_assert(!std::is_assignable_v<const cexa::tuple<MoveAssign>&, const cexa::tuple<MoveAssign>&>);

KOKKOS_INLINE_FUNCTION constexpr bool test() {
  // reference types
  {
    int i1 = 1;
    int i2 = 2;
    double d1 = 3.0;
    double d2 = 5.0;
    const cexa::tuple<int&, double&> t1{i1, d1};
    const cexa::tuple<int&, double&> t2{i2, d2};
    t2 = t1;
    CEXA_EXPECT_EQ(cexa::get<0>(t2), 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t2), 3.0);
  }

  // user defined const copy assignment
  {
    const cexa::tuple<ConstCopyAssign> t1{1};
    const cexa::tuple<ConstCopyAssign> t2{2};
    t2 = t1;
    CEXA_EXPECT_EQ(cexa::get<0>(t2).val, 1);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    cexa::tuple<TracedAssignment, const TracedAssignment> t1{};
    const cexa::tuple<TracedAssignment, const TracedAssignment> t2{};
    t2 = t1;
    CEXA_EXPECT_EQ(cexa::get<0>(t2).constCopyAssign, 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t2).constCopyAssign, 1);
  }

  return true;
}

// clang-format off
CEXA_TEST(tuple_assign, const_copy, (
  test();

  // FIXME: gcc cannot have mutable member in constant expression
#if !defined(KOKKOS_COMPILER_GNU) || KOKKOS_COMPILER_GNU >= 1300
  static_assert(test());
#endif
))
// clang-format on
#endif
