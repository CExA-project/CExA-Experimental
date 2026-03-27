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

// tuple& operator=(tuple&& u);

// UNSUPPORTED: c++03

#include <memory>
#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/MoveOnly.h>

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  KOKKOS_DEFAULTED_FUNCTION CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
};
static_assert(std::is_copy_assignable<CopyAssignable>::value, "");
struct MoveAssignable {
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  KOKKOS_DEFAULTED_FUNCTION MoveAssignable& operator=(MoveAssignable&&) = default;
};
struct NothrowMoveAssignable {
  KOKKOS_INLINE_FUNCTION NothrowMoveAssignable& operator=(NothrowMoveAssignable&&) noexcept { return *this; }
};
struct PotentiallyThrowingMoveAssignable {
  KOKKOS_INLINE_FUNCTION PotentiallyThrowingMoveAssignable& operator=(PotentiallyThrowingMoveAssignable&&) { return *this; }
};

#if defined(CEXA_ON_DEVICE)
__device__ int copied = 0;
__device__ int moved = 0;
#else
int copied = 0;
int moved = 0;
#endif

struct CountAssign {
  KOKKOS_INLINE_FUNCTION static void reset() { copied = moved = 0; }
  KOKKOS_DEFAULTED_FUNCTION CountAssign() = default;
  KOKKOS_INLINE_FUNCTION CountAssign& operator=(CountAssign const&) { ++copied; return *this; }
  KOKKOS_INLINE_FUNCTION CountAssign& operator=(CountAssign&&) { ++moved; return *this; }
};

KOKKOS_INLINE_FUNCTION constexpr
bool test()
{
    {
        typedef cexa::tuple<> T;
        T t0;
        [[maybe_unused]] T t;
        t = std::move(t0);
    }
    {
        typedef cexa::tuple<MoveOnly> T;
        T t0(MoveOnly(0));
        T t;
        t = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
    }
    {
        typedef cexa::tuple<MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1));
        T t;
        t = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1);
    }
    {
        typedef cexa::tuple<MoveOnly, MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
        T t;
        t = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 2);
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
        t = std::move(t2);
        CEXA_EXPECT_EQ(cexa::get<0>(t), x2);
        CEXA_EXPECT_EQ(&cexa::get<0>(t), &x);
        CEXA_EXPECT_EQ(cexa::get<1>(t), y2);
        CEXA_EXPECT_EQ(&cexa::get<1>(t), &y);
    }
    return true;
}

// clang-format off
CEXA_TEST(tuple_assign, move, (
    test();
    static_assert(test());

    {
        // test that the implicitly generated move assignment operator
        // is properly deleted
        using T = cexa::tuple<std::unique_ptr<int>>;
        static_assert(std::is_move_assignable<T>::value, "");
        static_assert(!std::is_copy_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<int, NonAssignable>;
        static_assert(!std::is_move_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<int, MoveAssignable>;
        static_assert(std::is_move_assignable<T>::value, "");
    }
    {
        // The move should decay to a copy.
        CountAssign::reset();
        using T = cexa::tuple<CountAssign, CopyAssignable>;
        static_assert(std::is_move_assignable<T>::value, "");
        T t1;
        T t2;
        t1 = std::move(t2);
        CEXA_EXPECT_EQ(copied, 1);
        CEXA_EXPECT_EQ(moved, 0);
    }
    {
        using T = cexa::tuple<int, NonAssignable>;
        static_assert(!std::is_move_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<int, MoveAssignable>;
        static_assert(std::is_move_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<NothrowMoveAssignable, int>;
        static_assert(std::is_nothrow_move_assignable<T>::value, "");
    }
    {
        using T = cexa::tuple<PotentiallyThrowingMoveAssignable, int>;
        static_assert(!std::is_nothrow_move_assignable<T>::value, "");
    }
    {
        // We assign through the reference and don't move out of the incoming ref,
        // so this doesn't work (but would if the type were CopyAssignable).
        using T1 = cexa::tuple<MoveAssignable&, int>;
        static_assert(!std::is_move_assignable<T1>::value, "");

        // ... works if it's CopyAssignable
        using T2 = cexa::tuple<CopyAssignable&, int>;
        static_assert(std::is_move_assignable<T2>::value, "");

        // For rvalue-references, we can move-assign if the type is MoveAssignable
        // or CopyAssignable (since in the worst case the move will decay into a copy).
        using T3 = cexa::tuple<MoveAssignable&&, int>;
        using T4 = cexa::tuple<CopyAssignable&&, int>;
        static_assert(std::is_move_assignable<T3>::value, "");
        static_assert(std::is_move_assignable<T4>::value, "");

        // In all cases, we can't move-assign if the types are not assignable,
        // since we assign through the reference.
        using T5 = cexa::tuple<NonAssignable&, int>;
        using T6 = cexa::tuple<NonAssignable&&, int>;
        static_assert(!std::is_move_assignable<T5>::value, "");
        static_assert(!std::is_move_assignable<T6>::value, "");
    }
))
// clang-format on
