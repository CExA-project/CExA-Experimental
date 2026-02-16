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

// template<class... Types>
//   tuple<Types&...> tie(Types&... t);

// UNSUPPORTED: c++03

#include <string>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>


KOKKOS_INLINE_FUNCTION constexpr bool test_tie()
{
    {
        int i = 42;
        double f = 1.1;
        using ExpectT = cexa::tuple<int&, decltype(cexa::ignore)&, double&>;
        auto res = cexa::tie(i, cexa::ignore, f);
        static_assert(std::is_same<ExpectT, decltype(res)>::value, "");
        CEXA_EXPECT_EQ(&cexa::get<0>(res), &i);
        CEXA_EXPECT_EQ(&cexa::get<1>(res), &cexa::ignore);
        CEXA_EXPECT_EQ(&cexa::get<2>(res), &f);

#if TEST_STD_VER >= 20
        res = cexa::make_tuple(101, nullptr, -1.0);
        CEXA_EXPECT_EQ(i, 101);
        CEXA_EXPECT_EQ(f, -1.0);
#endif
    }
    return true;
}

TEST(host_tuple_creation, tie_host) {
    int i = 0;
    std::string s;
    cexa::tie(i, cexa::ignore, s) = cexa::make_tuple(42, 3.14, "C++");
    CEXA_EXPECT_EQ(i, 42);
    CEXA_EXPECT_EQ(s, "C++");
}

// clang-format off
CEXA_TEST(tuple_creation, tie, (
    test_tie();
    static_assert(test_tie(), "");

))
// clang-format on
