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
//   typename tuple_element<I, tuple<Types...> >::type&
//   get(tuple<Types...>& t);

// UNSUPPORTED: c++03

#include <string>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>


struct Empty {};

struct S {
   cexa::tuple<int, Empty> a;
   int k;
   Empty e;
   KOKKOS_INLINE_FUNCTION constexpr S() : a{1,Empty{}}, k(cexa::get<0>(a)), e(cexa::get<1>(a)) {}
   };

KOKKOS_INLINE_FUNCTION constexpr cexa::tuple<int, int> getP () { return { 3, 4 }; }

TEST(host_tuple_elem, get_non_const_host) {
    {
        typedef cexa::tuple<std::string, int> T;
        T t("high", 5);
        CEXA_EXPECT_EQ(cexa::get<0>(t), "high");
        CEXA_EXPECT_EQ(cexa::get<1>(t), 5);
        cexa::get<0>(t) = "four";
        cexa::get<1>(t) = 4;
        CEXA_EXPECT_EQ(cexa::get<0>(t), "four");
        CEXA_EXPECT_EQ(cexa::get<1>(t), 4);
    }
    {
        typedef cexa::tuple<double&, std::string, int> T;
        double d = 1.5;
        T t(d, "high", 5);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1.5);
        CEXA_EXPECT_EQ(cexa::get<1>(t), "high");
        CEXA_EXPECT_EQ(cexa::get<2>(t), 5);
        cexa::get<0>(t) = 2.5;
        cexa::get<1>(t) = "four";
        cexa::get<2>(t) = 4;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2.5);
        CEXA_EXPECT_EQ(cexa::get<1>(t), "four");
        CEXA_EXPECT_EQ(cexa::get<2>(t), 4);
        CEXA_EXPECT_EQ(d, 2.5);
    }
}

// clang-format on
CEXA_TEST(tuple_elem, get_non_const, (
    {
        typedef cexa::tuple<int> T;
        T t(3);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 3);
        cexa::get<0>(t) = 2;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
    }
    {
        typedef cexa::tuple<float, int> T;
        T t(1.f, 5);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1.f);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 5);
        cexa::get<0>(t) = 2.f;
        cexa::get<1>(t) = 4;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2.f);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 4);
    }
    {
        typedef cexa::tuple<double&, float, int> T;
        double d = 1.5;
        T t(d, 1.f, 5);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1.5);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1.f);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 5);
        cexa::get<0>(t) = 2.5;
        cexa::get<1>(t) = 2.f;
        cexa::get<2>(t) = 4;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2.5);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 2.f);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 4);
        CEXA_EXPECT_EQ(d, 2.5);
    }
    { // get on an rvalue tuple
        static_assert ( cexa::get<0> ( cexa::make_tuple ( 0.0f, 1, 2.0, 3L )) == 0, "" );
        static_assert ( cexa::get<1> ( cexa::make_tuple ( 0.0f, 1, 2.0, 3L )) == 1, "" );
        static_assert ( cexa::get<2> ( cexa::make_tuple ( 0.0f, 1, 2.0, 3L )) == 2, "" );
        static_assert ( cexa::get<3> ( cexa::make_tuple ( 0.0f, 1, 2.0, 3L )) == 3, "" );
        static_assert(S().k == 1, "");
        static_assert(cexa::get<1>(getP()) == 4, "");
    }
))
// clang-format on
