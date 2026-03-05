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

// UNSUPPORTED: c++03, c++11

#include <utility>
#include <memory>
#include <string>
#include <complex>
#include <type_traits>

#include <Kokkos_Complex.hpp>
#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

TEST(host_tuple_elem, tuple_by_type_host) {
    typedef std::complex<float> cf;
    {
    auto t1 = cexa::tuple<int, std::string, cf> { 42, "Hi", { 1,2 }};
    CEXA_EXPECT_EQ(cexa::get<int>(t1), 42); // find at the beginning
    CEXA_EXPECT_EQ(cexa::get<std::string>(t1), "Hi"); // find in the middle
    CEXA_EXPECT_EQ(cexa::get<cf>(t1).real(), 1); // find at the end
    CEXA_EXPECT_EQ(cexa::get<cf>(t1).imag(), 2);
    }

    {
    auto t2 = cexa::tuple<int, std::string, int, cf> { 42, "Hi", 23, { 1,2 }};
//  get<int> would fail!
    CEXA_EXPECT_EQ(cexa::get<std::string>(t2), "Hi");
    CEXA_EXPECT_EQ(cexa::get<cf>(t2), (cf{ 1,2 }));
    }

    {
    typedef std::unique_ptr<int> upint;
    cexa::tuple<upint> t(upint(new int(4)));
    upint p = cexa::get<upint>(std::move(t)); // get rvalue
    CEXA_EXPECT_EQ(*p, 4);
    CEXA_EXPECT_EQ(cexa::get<upint>(t), nullptr); // has been moved from
    }

    {
    typedef std::unique_ptr<int> upint;
    const cexa::tuple<upint> t(upint(new int(4)));
    const upint&& p = cexa::get<upint>(std::move(t)); // get const rvalue
    CEXA_EXPECT_EQ(*p, 4);
    CEXA_EXPECT(cexa::get<upint>(t) != nullptr);
    }

}

// clang-format off
CEXA_TEST(tuple_elem, tuple_by_type, (
    typedef Kokkos::complex<float> cf;
    {
    auto t1 = cexa::tuple<int, float, cf> { 42, 1.f, { 1,2 }};
    CEXA_EXPECT_EQ(cexa::get<int>(t1), 42); // find at the beginning
    CEXA_EXPECT_EQ(cexa::get<float>(t1), 1.f); // find in the middle
    CEXA_EXPECT_EQ(cexa::get<cf>(t1).real(), 1); // find at the end
    CEXA_EXPECT_EQ(cexa::get<cf>(t1).imag(), 2);
    }

    {
    auto t2 = cexa::tuple<int, float, int, cf> { 42, 1.f, 23, { 1,2 }};
//  get<int> would fail!
    CEXA_EXPECT_EQ(cexa::get<float>(t2), 1.f);
    CEXA_EXPECT_EQ(cexa::get<cf>(t2), (cf{ 1,2 }));
    }
    {
    constexpr cexa::tuple<int, const int, double, double> p5 { 1, 2, 3.4, 5.6 };
    static_assert ( cexa::get<int>(p5) == 1, "" );
    static_assert ( cexa::get<const int>(p5) == 2, "" );
    }

    {
    const cexa::tuple<int, const int, double, double> p5 { 1, 2, 3.4, 5.6 };
    const int &i1 = cexa::get<int>(p5);
    const int &i2 = cexa::get<const int>(p5);
    CEXA_EXPECT_EQ(i1, 1);
    CEXA_EXPECT_EQ(i2, 2);
    }

    {
    int x = 42;
    int y = 43;
    cexa::tuple<int&, int const&> const t(x, y);
    static_assert(std::is_same<int&, decltype(cexa::get<int&>(std::move(t)))>::value, "");
    static_assert(noexcept(cexa::get<int&>(std::move(t))), "");
    static_assert(std::is_same<int const&, decltype(cexa::get<int const&>(std::move(t)))>::value, "");
    static_assert(noexcept(cexa::get<int const&>(std::move(t))), "");
    }

    {
    int x = 42;
    int y = 43;
    cexa::tuple<int&&, int const&&> const t(std::move(x), std::move(y));
    static_assert(std::is_same<int&&, decltype(cexa::get<int&&>(std::move(t)))>::value, "");
    static_assert(noexcept(cexa::get<int&&>(std::move(t))), "");
    static_assert(std::is_same<int const&&, decltype(cexa::get<int const&&>(std::move(t)))>::value, "");
    static_assert(noexcept(cexa::get<int const&&>(std::move(t))), "");
    }

    {
    constexpr const cexa::tuple<int, const int, double, double> t { 1, 2, 3.4, 5.6 };
    static_assert(cexa::get<int>(std::move(t)) == 1, "");
    static_assert(cexa::get<const int>(std::move(t)) == 2, "");
    }
))
// clang-format off
