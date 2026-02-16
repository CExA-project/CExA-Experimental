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

// template <size_t I, class... Types>
//   typename tuple_element<I, tuple<Types...> >::type const&
//   get(const tuple<Types...>& t);

// UNSUPPORTED: c++03

#include <string>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct Empty {};

TEST(host_tuple_elem, get_const_host) {
    {
        typedef cexa::tuple<std::string, int> T;
        const T t("high", 5);
        CEXA_EXPECT_EQ(cexa::get<0>(t), "high");
        CEXA_EXPECT_EQ(cexa::get<1>(t), 5);
    }
    {
        typedef cexa::tuple<double&, std::string, int> T;
        double d = 1.5;
        const T t(d, "high", 5);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1.5);
        CEXA_EXPECT_EQ(cexa::get<1>(t), "high");
        CEXA_EXPECT_EQ(cexa::get<2>(t), 5);
        cexa::get<0>(t) = 2.5;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2.5);
        CEXA_EXPECT_EQ(cexa::get<1>(t), "high");
        CEXA_EXPECT_EQ(cexa::get<2>(t), 5);
        CEXA_EXPECT_EQ(d, 2.5);
    }
}

// clang-format off
CEXA_TEST(tuple_elem, get_const, (
    {
        typedef cexa::tuple<int> T;
        const T t(3);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 3);
    }
    {
        typedef cexa::tuple<double, int> T;
        constexpr T t(2.718, 5);
        static_assert(cexa::get<0>(t) == 2.718, "");
        static_assert(cexa::get<1>(t) == 5, "");
    }
    {
        typedef cexa::tuple<Empty> T;
        constexpr T t{Empty()};
        [[maybe_unused]] constexpr Empty e = cexa::get<0>(t);
    }
    {
        typedef cexa::tuple<double&, float, int> T;
        double d = 1.5;
        const T t(d, 1.f, 5);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1.5);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1.f);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 5);
        cexa::get<0>(t) = 2.5;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2.5);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1.f);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 5);
        CEXA_EXPECT_EQ(d, 2.5);
    }
))
// clang-format on
