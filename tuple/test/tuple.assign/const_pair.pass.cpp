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

// template <class U1, class U2>
//   tuple& operator=(const pair<U1, U2>& u);

// UNSUPPORTED: c++03

#include <memory>
#include <type_traits>
#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct NothrowCopyAssignable {
    NothrowCopyAssignable(NothrowCopyAssignable const&) = delete;
    NothrowCopyAssignable& operator=(NothrowCopyAssignable const&) noexcept { return *this; }
};
struct PotentiallyThrowingCopyAssignable {
    PotentiallyThrowingCopyAssignable(PotentiallyThrowingCopyAssignable const&) = delete;
    PotentiallyThrowingCopyAssignable& operator=(PotentiallyThrowingCopyAssignable const&) { return *this; }
};

// TODO: add a kokkos pair version
constexpr bool test()
{
    {
        typedef std::pair<long, char> T0;
        typedef cexa::tuple<long long, short> T1;
        T0 t0(2, 'a');
        T1 t1;
        t1 = t0;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), short('a'));
    }
    return true;
}

// clang-format off
TEST(host_tuple_assign, const_pair) {
    test();
    static_assert(test());

    {
        // test that the implicitly generated copy assignment operator
        // is properly deleted
        using T = cexa::tuple<int, int>;
        using P = cexa::tuple<std::unique_ptr<int>, std::unique_ptr<int>>;
        static_assert(!std::is_assignable<T&, const P &>::value, "");
    }
    {
        typedef cexa::tuple<NothrowCopyAssignable, long> Tuple;
        typedef std::pair<NothrowCopyAssignable, int> Pair;
        static_assert(std::is_nothrow_assignable<Tuple&, Pair const&>::value, "");
        static_assert(std::is_nothrow_assignable<Tuple&, Pair&>::value, "");
        static_assert(std::is_nothrow_assignable<Tuple&, Pair const&&>::value, "");
    }
    {
        typedef cexa::tuple<PotentiallyThrowingCopyAssignable, long> Tuple;
        typedef std::pair<PotentiallyThrowingCopyAssignable, int> Pair;
        static_assert(std::is_assignable<Tuple&, Pair const&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Pair const&>::value, "");

        static_assert(std::is_assignable<Tuple&, Pair&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Pair&>::value, "");

        static_assert(std::is_assignable<Tuple&, Pair const&&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Pair const&&>::value, "");
    }
}
// clang-format on
