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

// tuple& operator=(const tuple& u);

// UNSUPPORTED: c++03

#include <memory>
#include <string>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable &&) = delete;
};
static_assert(std::is_copy_assignable<CopyAssignable>::value, "");
struct MoveAssignable {
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};
struct NothrowCopyAssignable {
  NothrowCopyAssignable& operator=(NothrowCopyAssignable const&) noexcept { return *this; }
};
struct PotentiallyThrowingCopyAssignable {
  PotentiallyThrowingCopyAssignable& operator=(PotentiallyThrowingCopyAssignable const&) { return *this; }
};

struct CopyAssignableInt {
  CopyAssignableInt& operator=(int&) { return *this; }
};

constexpr
bool test()
{
    {
        typedef cexa::tuple<> T;
        T t0;
        T t;
        t = t0;
    }
    {
        typedef cexa::tuple<int> T;
        T t0(2);
        T t;
        t = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
    }
    {
        typedef cexa::tuple<int, char> T;
        T t0(2, 'a');
        T t;
        t = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 'a');
    }
    {
        // test reference assignment.
        using T = cexa::tuple<int&, int&&>;
        int x = 42;
        int y = 100;
        int x2 = -1;
        int y2 = 500;
        T t(x, std::move(y));
        T t2(x2, std::move(y2));
        t = t2;
        CEXA_EXPECT_EQ(cexa::get<0>(t), x2);
        CEXA_EXPECT_EQ(&cexa::get<0>(t), &x);
        CEXA_EXPECT_EQ(cexa::get<1>(t), y2);
        CEXA_EXPECT_EQ(&cexa::get<1>(t), &y);
    }

    return true;
}

TEST(tuple_assign, copy_host) {
    // cannot be constexpr because of std::string
    typedef cexa::tuple<int, char, std::string> T;
    const T t0(2, 'a', "some text");
    T t;
    t = t0;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
    CEXA_EXPECT_EQ(cexa::get<1>(t), 'a');
    CEXA_EXPECT_EQ(cexa::get<2>(t), "some text");
}

// clang-format off
CEXA_TEST(tuple_assign, copy, (
    test();
    static_assert(test());

    {
        // test that the implicitly generated copy assignment operator
        // is properly deleted
        using T = cexa::tuple<std::unique_ptr<int>>;
        static_assert(!std::is_copy_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<int, NonAssignable>;
        static_assert(!std::is_copy_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<int, CopyAssignable>;
        static_assert(std::is_copy_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<int, MoveAssignable>;
        static_assert(!std::is_copy_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<int, int, int>;
        using P = std::pair<int, int>;
        static_assert(!std::is_assignable<T&, P>::value, "");
    }
    {
        // test const requirement
        using T = cexa::tuple<CopyAssignableInt, CopyAssignableInt>;
        using P = std::pair<int, int>;
        static_assert(!std::is_assignable<T&, P const>::value, "");
    }
    {
        using T = cexa::tuple<int, MoveAssignable>;
        using P = std::pair<int, MoveAssignable>;
        static_assert(!std::is_assignable<T&, P&>::value, "");
    }
    {
        using T = cexa::tuple<NothrowCopyAssignable, int>;
        static_assert(std::is_nothrow_copy_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<PotentiallyThrowingCopyAssignable, int>;
        static_assert(std::is_copy_assignable<T>::value, "");
        static_assert(!std::is_nothrow_copy_assignable<T>::value, "");
    }
))
// clang-format on
