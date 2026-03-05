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

// template<class... Types>
//     tuple<Types&&...> forward_as_tuple(Types&&... t);

// UNSUPPORTED: c++03

#include <type_traits>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

template <class Tuple>
KOKKOS_INLINE_FUNCTION void
test0(const Tuple&)
{
    static_assert(cexa::tuple_size<Tuple>::value == 0, "");
}

template <class Tuple>
KOKKOS_INLINE_FUNCTION void
test1a(const Tuple& t)
{
    static_assert(cexa::tuple_size<Tuple>::value == 1, "");
    static_assert(std::is_same<typename cexa::tuple_element<0, Tuple>::type, int&&>::value, "");
    CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
}

template <class Tuple>
KOKKOS_INLINE_FUNCTION void
test1b(const Tuple& t)
{
    static_assert(cexa::tuple_size<Tuple>::value == 1, "");
    static_assert(std::is_same<typename cexa::tuple_element<0, Tuple>::type, int&>::value, "");
    CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
}

template <class Tuple>
KOKKOS_INLINE_FUNCTION void
test2a(const Tuple& t)
{
    static_assert(cexa::tuple_size<Tuple>::value == 2, "");
    static_assert(std::is_same<typename cexa::tuple_element<0, Tuple>::type, double&>::value, "");
    static_assert(std::is_same<typename cexa::tuple_element<1, Tuple>::type, char&>::value, "");
    CEXA_EXPECT_EQ(cexa::get<0>(t), 2.5);
    CEXA_EXPECT_EQ(cexa::get<1>(t), 'a');
}

template <class Tuple>
KOKKOS_INLINE_FUNCTION constexpr int
test3(const Tuple&)
{
    return cexa::tuple_size<Tuple>::value;
}

// clang-format off
CEXA_TEST(tuple_creation, forward_as_tuple, (
    {
        test0(cexa::forward_as_tuple());
    }
    {
        test1a(cexa::forward_as_tuple(1));
    }
    {
        int i = 2;
        test1b(cexa::forward_as_tuple(i));
    }
    {
        double i = 2.5;
        char c = 'a';
        test2a(cexa::forward_as_tuple(i, c));
        static_assert ( test3 (cexa::forward_as_tuple(i, c)) == 2, "" );
    }
))
// clang-format on
