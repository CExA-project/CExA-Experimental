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
//   void swap(tuple<Types...>& x, tuple<Types...>& y);

// UNSUPPORTED: c++03

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/MoveOnly.h>

// clang-format off
CEXA_TEST(tuple_special, non_member_swap, (
    {
        typedef cexa::tuple<> T;
        T t0;
        T t1;
        swap(t0, t1);
    }
    {
        typedef cexa::tuple<MoveOnly> T;
        T t0(MoveOnly(0));
        T t1(MoveOnly(1));
        swap(t0, t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t0), 1);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 0);
    }
    {
        typedef cexa::tuple<MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1));
        T t1(MoveOnly(2), MoveOnly(3));
        swap(t0, t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t0), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t0), 3);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), 1);
    }
    {
        typedef cexa::tuple<MoveOnly, MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
        T t1(MoveOnly(3), MoveOnly(4), MoveOnly(5));
        swap(t0, t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t0), 3);
        CEXA_EXPECT_EQ(cexa::get<1>(t0), 4);
        CEXA_EXPECT_EQ(cexa::get<2>(t0), 5);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), 1);
        CEXA_EXPECT_EQ(cexa::get<2>(t1), 2);
    }
))
// clang-format on
