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

// template <class... UTypes> tuple(const tuple<UTypes...>& u);

// UNSUPPORTED: c++03

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct Explicit {
  int value;
  KOKKOS_INLINE_FUNCTION explicit Explicit(int x) : value(x) {}
};

struct Implicit {
  int value;
  KOKKOS_INLINE_FUNCTION Implicit(int x) : value(x) {}
};

struct ExplicitTwo {
    KOKKOS_INLINE_FUNCTION ExplicitTwo() {}
    KOKKOS_INLINE_FUNCTION ExplicitTwo(ExplicitTwo const&) {}
    KOKKOS_INLINE_FUNCTION ExplicitTwo(ExplicitTwo &&) {}

    template <class T, class = typename std::enable_if<!std::is_same<T, ExplicitTwo>::value>::type>
    KOKKOS_INLINE_FUNCTION explicit ExplicitTwo(T) {}
};

struct B
{
    int id_;

    KOKKOS_INLINE_FUNCTION explicit B(int i) : id_(i) {}
};

struct D
    : B
{
    KOKKOS_INLINE_FUNCTION explicit D(int i) : B(i) {}
};

struct A
{
    int id_;

    KOKKOS_INLINE_FUNCTION constexpr A(int i) : id_(i) {}
    KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(const A& x, const A& y) {return x.id_ == y.id_;}
};

struct C
{
    int id_;

    KOKKOS_INLINE_FUNCTION constexpr explicit C(int i) : id_(i) {}
    KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(const C& x, const C& y) {return x.id_ == y.id_;}
};

// clang-format off
CEXA_TEST(tuple_cnstr, convert_copy_pass, (
    {
        typedef cexa::tuple<long> T0;
        typedef cexa::tuple<long long> T1;
        T0 t0(2);
        T1 t1 = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
    }
    {
        typedef cexa::tuple<int> T0;
        typedef cexa::tuple<A> T1;
        constexpr T0 t0(2);
        constexpr T1 t1 = t0;
        static_assert(cexa::get<0>(t1) == 2, "");
    }
    {
        typedef cexa::tuple<int> T0;
        typedef cexa::tuple<C> T1;
        constexpr T0 t0(2);
        constexpr T1 t1{t0};
        static_assert(cexa::get<0>(t1) == C(2), "");
    }
    {
        typedef cexa::tuple<long, char> T0;
        typedef cexa::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1 = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
    }
    {
        typedef cexa::tuple<long, char, D> T0;
        typedef cexa::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1 = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
        CEXA_EXPECT_EQ(cexa::get<2>(t1).id_, 3);
    }
    {
        D d(3);
        typedef cexa::tuple<long, char, D&> T0;
        typedef cexa::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d);
        T1 t1 = t0;
        d.id_ = 2;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
        CEXA_EXPECT_EQ(cexa::get<2>(t1).id_, 2);
    }
    {
        typedef cexa::tuple<long, char, int> T0;
        typedef cexa::tuple<long long, int, B> T1;
        T0 t0(2, 'a', 3);
        T1 t1(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
        CEXA_EXPECT_EQ(cexa::get<2>(t1).id_, 3);
    }
    {
        const cexa::tuple<int> t1(42);
        cexa::tuple<Explicit> t2(t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t2).value, 42);
    }
    {
        const cexa::tuple<int> t1(42);
        cexa::tuple<Implicit> t2 = t1;
        CEXA_EXPECT_EQ(cexa::get<0>(t2).value, 42);
    }
    {
        // static_assert(std::is_convertible<ExplicitTwo&&, ExplicitTwo>::value, "");
        // static_assert(std::is_convertible<cexa::tuple<ExplicitTwo&&>&&, const cexa::tuple<ExplicitTwo>&>::value, "");
        //
        // ExplicitTwo e;
        // cexa::tuple<ExplicitTwo> t = cexa::tuple<ExplicitTwo&&>(std::move(e));
        // ((void)t);
    }
))
// clang-format off
