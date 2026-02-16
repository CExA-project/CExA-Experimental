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
//   const typename tuple_element<I, tuple<Types...> >::type&&
//   get(const tuple<Types...>&& t);

// UNSUPPORTED: c++03

#include <utility>
#include <string>
#include <type_traits>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

TEST(host_tuple_elem, get_const_rv_host) {
    {
    typedef cexa::tuple<std::string, int> T;
    const T t("high", 5);
    static_assert(std::is_same<const std::string&&, decltype(cexa::get<0>(std::move(t)))>::value, "");
    static_assert(noexcept(cexa::get<0>(std::move(t))), "");
    static_assert(std::is_same<const int&&, decltype(cexa::get<1>(std::move(t)))>::value, "");
    static_assert(noexcept(cexa::get<1>(std::move(t))), "");
    const std::string&& s = cexa::get<0>(std::move(t));
    const int&& i = cexa::get<1>(std::move(t));
    CEXA_EXPECT_EQ(s, "high");
    CEXA_EXPECT_EQ(i, 5);
    }
}

// clang-format off
CEXA_TEST(tuple_elem, get_const_rv, (
    {
    typedef cexa::tuple<int> T;
    const T t(3);
    static_assert(std::is_same<const int&&, decltype(cexa::get<0>(std::move(t)))>::value, "");
    static_assert(noexcept(cexa::get<0>(std::move(t))), "");
    const int&& i = cexa::get<0>(std::move(t));
    CEXA_EXPECT_EQ(i, 3);
    }

    {
    int x = 42;
    int const y = 43;
    cexa::tuple<int&, int const&> const p(x, y);
    static_assert(std::is_same<int&, decltype(cexa::get<0>(std::move(p)))>::value, "");
    static_assert(noexcept(cexa::get<0>(std::move(p))), "");
    static_assert(std::is_same<int const&, decltype(cexa::get<1>(std::move(p)))>::value, "");
    static_assert(noexcept(cexa::get<1>(std::move(p))), "");
    }

    {
    int x = 42;
    int const y = 43;
    cexa::tuple<int&&, int const&&> const p(std::move(x), std::move(y));
    static_assert(std::is_same<int&&, decltype(cexa::get<0>(std::move(p)))>::value, "");
    static_assert(noexcept(cexa::get<0>(std::move(p))), "");
    static_assert(std::is_same<int const&&, decltype(cexa::get<1>(std::move(p)))>::value, "");
    static_assert(noexcept(cexa::get<1>(std::move(p))), "");
    }

    {
    typedef cexa::tuple<double, int> T;
    constexpr const T t(2.718, 5);
    static_assert(cexa::get<0>(std::move(t)) == 2.718, "");
    static_assert(cexa::get<1>(std::move(t)) == 5, "");
    }
))
// clang-format on
