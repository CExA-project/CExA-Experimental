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

// tuple(const tuple& u) = default;

// UNSUPPORTED: c++03

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct Empty {};

// clang-format off
CEXA_TEST(tuple_cnstr, copy, (
    {
        typedef cexa::tuple<> T;
        T t0;
        [[maybe_unused]] T t = t0;
        // ((void)t); // Prevent unused warning
    }
    {
        typedef cexa::tuple<int> T;
        T t0(2);
        T t = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
    }
    {
        typedef cexa::tuple<int, char> T;
        T t0(2, 'a');
        T t = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 'a');
    }
    {
        // typedef cexa::tuple<int, char, std::string> T;
        typedef cexa::tuple<int, char, Kokkos::pair<int, float>> T;
        const T t0(2, 'a', {2, 6.7});
        T t = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 'a');
        CEXA_EXPECT_EQ(cexa::get<2>(t), (Kokkos::pair<int, float>{2, 6.7}));
    }
    {
        typedef cexa::tuple<int> T;
        constexpr T t0(2);
        constexpr T t = t0;
        static_assert(cexa::get<0>(t) == 2, "");
    }
    {
        typedef cexa::tuple<Empty> T;
        constexpr T t0;
        constexpr T t = t0;
        [[maybe_unused]] constexpr Empty e = cexa::get<0>(t);
        // ((void)e); // Prevent unused warning
    }
))
// clang-format off
