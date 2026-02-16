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

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

// UNSUPPORTED: c++03

#include <type_traits>

#include <Kokkos_Macros.hpp>
#include <tuple.hpp>
#include <gtest/gtest.h>
#include <support/cexa_test_macros.hpp>
#include <support/MoveOnly.h>

struct Empty {};
struct A {
  int id_;
  KOKKOS_INLINE_FUNCTION explicit constexpr A(int i) : id_(i) {}
};

struct NoDefault {
  NoDefault() = delete;
};

// Make sure the _Up... constructor SFINAEs out when there are fewer
// constructor arguments than tuple elements.
namespace test1 {
typedef cexa::tuple<MoveOnly, NoDefault> Tuple;

static_assert(!std::is_constructible<Tuple, MoveOnly>::value, "");

static_assert(std::is_constructible<Tuple, MoveOnly, NoDefault>::value, "");
}  // namespace test1

namespace test2 {
typedef cexa::tuple<MoveOnly, MoveOnly, NoDefault> Tuple;

static_assert(!std::is_constructible<Tuple, MoveOnly, MoveOnly>::value, "");

static_assert(
    std::is_constructible<Tuple, MoveOnly, MoveOnly, NoDefault>::value, "");
}  // namespace test2

namespace test3 {
// Same idea as above but with a nested tuple type.
typedef cexa::tuple<MoveOnly, NoDefault> Tuple;
typedef cexa::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly> NestedTuple;

static_assert(!std::is_constructible<NestedTuple, MoveOnly, MoveOnly, MoveOnly,
                                     MoveOnly>::value,
              "");

static_assert(std::is_constructible<NestedTuple, MoveOnly, Tuple, MoveOnly,
                                    MoveOnly>::value,
              "");
}  // namespace test3

namespace test4 {
constexpr cexa::tuple<A, A> t(3, 2);
static_assert(cexa::get<0>(t).id_ == 3, "");
}  // namespace test4

// clang-format off
CEXA_TEST(tuple_cnstr, UTypes, (
    {
        cexa::tuple<MoveOnly> t(MoveOnly(0));
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
    }
    {
        cexa::tuple<MoveOnly, MoveOnly> t(MoveOnly(0), MoveOnly(1));
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1);
    }
    {
        cexa::tuple<MoveOnly, MoveOnly, MoveOnly> t(MoveOnly(0),
                                                   MoveOnly(1),
                                                   MoveOnly(2));
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 2);
    }
    {
        constexpr cexa::tuple<Empty> t0{Empty()};
        (void)t0;
    }
))
// clang-format on
