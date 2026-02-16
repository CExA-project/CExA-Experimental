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

// template <class... UTypes>
//   tuple& operator=(const tuple<UTypes...>& u);

// UNSUPPORTED: c++03

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct B {
    int id_;

    KOKKOS_INLINE_FUNCTION constexpr explicit B(int i = 0) : id_(i) {}
};

struct D : B {
    KOKKOS_INLINE_FUNCTION constexpr explicit D(int i = 0) : B(i) {}
};

struct NonAssignable {
    NonAssignable& operator=(NonAssignable const&) = delete;
    NonAssignable& operator=(NonAssignable&&) = delete;
};

struct NothrowCopyAssignable {
    NothrowCopyAssignable(NothrowCopyAssignable const&) = delete;
    KOKKOS_INLINE_FUNCTION NothrowCopyAssignable& operator=(NothrowCopyAssignable const&) noexcept { return *this; }
};

struct PotentiallyThrowingCopyAssignable {
    PotentiallyThrowingCopyAssignable(PotentiallyThrowingCopyAssignable const&) = delete;
    KOKKOS_INLINE_FUNCTION PotentiallyThrowingCopyAssignable& operator=(PotentiallyThrowingCopyAssignable const&) { return *this; }
};

KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX20
bool test()
{
    {
        typedef cexa::tuple<long> T0;
        typedef cexa::tuple<long long> T1;
        T0 t0(2);
        T1 t1;
        t1 = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
    }
    {
        typedef cexa::tuple<long, char> T0;
        typedef cexa::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1;
        t1 = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
    }
    {
        typedef cexa::tuple<long, char, D> T0;
        typedef cexa::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1;
        t1 = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
        CEXA_EXPECT_EQ(cexa::get<2>(t1).id_, 3);
    }
    {
        D d(3);
        D d2(2);
        typedef cexa::tuple<long, char, D&> T0;
        typedef cexa::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d2);
        T1 t1(1, 'b', d);
        t1 = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
        CEXA_EXPECT_EQ(cexa::get<2>(t1).id_, 2);
    }
    {
        // Test that tuple evaluates correctly applies an lvalue reference
        // before evaluating is_assignable (i.e. 'is_assignable<int&, int&>')
        // instead of evaluating 'is_assignable<int&&, int&>' which is false.
        int x = 42;
        int y = 43;
        cexa::tuple<int&&> t(std::move(x));
        cexa::tuple<int&> t2(y);
        t = t2;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 43);
        CEXA_EXPECT_EQ(&cexa::get<0>(t), &x);
    }
    return true;
}

// clang-format off
CEXA_TEST(tuple_assign, convert_copy, (
    test();
#if TEST_STD_VER >= 20
    static_assert(test());
#endif

    {
        using T = cexa::tuple<int, NonAssignable>;
        using U = cexa::tuple<NonAssignable, int>;
        static_assert(!std::is_assignable<T&, U const&>::value, "");
        static_assert(!std::is_assignable<U&, T const&>::value, "");
    }
    {
        typedef cexa::tuple<NothrowCopyAssignable, long> T0;
        typedef cexa::tuple<NothrowCopyAssignable, int> T1;
        static_assert(std::is_nothrow_assignable<T0&, T1 const&>::value, "");
    }
    {
        typedef cexa::tuple<PotentiallyThrowingCopyAssignable, long> T0;
        typedef cexa::tuple<PotentiallyThrowingCopyAssignable, int> T1;
        static_assert(std::is_assignable<T0&, T1 const&>::value, "");
        static_assert(!std::is_nothrow_assignable<T0&, T1 const&>::value, "");
    }
))
// clang-format on
