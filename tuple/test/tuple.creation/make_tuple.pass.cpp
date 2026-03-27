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
//   tuple<VTypes...> make_tuple(Types&&... t);

// UNSUPPORTED: c++03

#include <functional>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

template<class T>
struct Ref {
    T& value;

    KOKKOS_INLINE_FUNCTION constexpr Ref(const Ref& other) : value(other.value) {}
    KOKKOS_INLINE_FUNCTION constexpr Ref(Ref&& other) : value(other.value) {}
    KOKKOS_INLINE_FUNCTION constexpr Ref(T& other) : value(other) {}

    KOKKOS_INLINE_FUNCTION constexpr Ref& operator=(T other) {
      value = other;
      return *this;
    }
    KOKKOS_INLINE_FUNCTION constexpr operator T&() { return value; }
};

KOKKOS_INLINE_FUNCTION constexpr
bool device_test()
{
    int i = 0;
    float j = 0;
    cexa::tuple<int, Ref<int>, Ref<float>> t =
        cexa::make_tuple(1, Ref{i}, Ref{j});
    CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t), 0);
    CEXA_EXPECT_EQ(cexa::get<2>(t), 0);
    i = 2;
    j = 3.5;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t), 2);
    CEXA_EXPECT_EQ(cexa::get<2>(t), 3.5);
    cexa::get<1>(t) = 0;
    cexa::get<2>(t) = 0;
    CEXA_EXPECT_EQ(i, 0);
    CEXA_EXPECT_EQ(j, 0);

    return true;
}

constexpr
bool test()
{
    int i = 0;
    float j = 0;
    cexa::tuple<int, int&, float&> t =
        cexa::make_tuple(1, std::ref(i), std::ref(j));
    CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t), 0);
    CEXA_EXPECT_EQ(cexa::get<2>(t), 0);
    i = 2;
    j = 3.5;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
    CEXA_EXPECT_EQ(cexa::get<1>(t), 2);
    CEXA_EXPECT_EQ(cexa::get<2>(t), 3.5);
    cexa::get<1>(t) = 0;
    cexa::get<2>(t) = 0;
    CEXA_EXPECT_EQ(i, 0);
    CEXA_EXPECT_EQ(j, 0);

    return true;
}

TEST(host_tuple_creation, make_tuple_host) {
    test();
    static_assert(test());
}

// clang-format off
CEXA_TEST(tuple_creation, make_tuple, (
    device_test();
    static_assert(device_test());

    {
        constexpr auto t1 = cexa::make_tuple(0, 1, 3.14);
        constexpr int i1 = cexa::get<1>(t1);
        constexpr double d1 = cexa::get<2>(t1);
        static_assert (i1 == 1, "" );
        static_assert (d1 == 3.14, "" );
    }
))
// clang-format on
