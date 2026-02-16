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
//   tuple& operator=(tuple<UTypes...>&& u);

// UNSUPPORTED: c++03

#include <memory>
#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct B {
    int id_;
    KOKKOS_INLINE_FUNCTION explicit B(int i = 0) : id_(i) {}
    KOKKOS_DEFAULTED_FUNCTION B(const B&) = default;
    KOKKOS_DEFAULTED_FUNCTION B& operator=(const B&) = default;
    KOKKOS_INLINE_FUNCTION virtual ~B() {}
};

struct D : B {
    KOKKOS_INLINE_FUNCTION explicit D(int i) : B(i) {}
};

struct E {
  KOKKOS_DEFAULTED_FUNCTION constexpr E() = default;
  KOKKOS_INLINE_FUNCTION constexpr E& operator=(int) {
      return *this;
  }
};

struct NothrowMoveAssignable {
    KOKKOS_INLINE_FUNCTION NothrowMoveAssignable& operator=(NothrowMoveAssignable&&) noexcept { return *this; }
};

struct PotentiallyThrowingMoveAssignable {
    KOKKOS_INLINE_FUNCTION PotentiallyThrowingMoveAssignable& operator=(PotentiallyThrowingMoveAssignable&&) { return *this; }
};

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};

struct MoveAssignable {
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  KOKKOS_DEFAULTED_FUNCTION MoveAssignable& operator=(MoveAssignable&&) = default;
};

struct CopyAssignable {
  KOKKOS_DEFAULTED_FUNCTION CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
};

struct TrackMove
{
    KOKKOS_INLINE_FUNCTION TrackMove() : value(0), moved_from(false) { }
    KOKKOS_INLINE_FUNCTION explicit TrackMove(int v) : value(v), moved_from(false) { }
    KOKKOS_INLINE_FUNCTION TrackMove(TrackMove const& other) : value(other.value), moved_from(false) { }
    KOKKOS_INLINE_FUNCTION TrackMove(TrackMove&& other) : value(other.value), moved_from(false) {
        other.moved_from = true;
    }
    KOKKOS_INLINE_FUNCTION TrackMove& operator=(TrackMove const& other) {
        value = other.value;
        moved_from = false;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION TrackMove& operator=(TrackMove&& other) {
        value = other.value;
        moved_from = false;
        other.moved_from = true;
        return *this;
    }

    int value;
    bool moved_from;
};

KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX20
bool test()
{
    {
        typedef cexa::tuple<long> T0;
        typedef cexa::tuple<long long> T1;
        T0 t0(2);
        T1 t1;
        t1 = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
    }
    {
        typedef cexa::tuple<long, char> T0;
        typedef cexa::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1;
        t1 = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
    }
    {
        // Test that tuple evaluates correctly applies an lvalue reference
        // before evaluating is_assignable (i.e. 'is_assignable<int&, int&&>')
        // instead of evaluating 'is_assignable<int&&, int&&>' which is false.
        int x = 42;
        int y = 43;
        cexa::tuple<int&&, E> t(std::move(x), E{});
        cexa::tuple<int&&, int> t2(std::move(y), 44);
        t = std::move(t2);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 43);
        CEXA_EXPECT_EQ(&cexa::get<0>(t), &x);
    }

    return true;
}

TEST(tuple_assign, convert_move_host) {
    typedef cexa::tuple<long, char, std::unique_ptr<D>> T0;
    typedef cexa::tuple<long long, int, std::unique_ptr<B>> T1;
    T0 t0(2, 'a', std::unique_ptr<D>(new D(3)));
    T1 t1;
    t1 = std::move(t0);
    CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
    CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
    CEXA_EXPECT_EQ(cexa::get<2>(t1)->id_, 3);
}

// clang-format off
CEXA_TEST(tuple_assign, convert_move, (
    test();
#if TEST_STD_VER >= 20
    static_assert(test());
#endif

    {
        typedef cexa::tuple<long, char, D> T0;
        typedef cexa::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1;
        t1 = std::move(t0);
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
        t1 = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
        CEXA_EXPECT_EQ(cexa::get<2>(t1).id_, 2);
    }

    {
        using T = cexa::tuple<int, NonAssignable>;
        using U = cexa::tuple<NonAssignable, int>;
        static_assert(!std::is_assignable<T&, U&&>::value, "");
        static_assert(!std::is_assignable<U&, T&&>::value, "");
    }
    {
        typedef cexa::tuple<NothrowMoveAssignable, long> T0;
        typedef cexa::tuple<NothrowMoveAssignable, int> T1;
        static_assert(std::is_nothrow_assignable<T0&, T1&&>::value, "");
    }
    {
        typedef cexa::tuple<PotentiallyThrowingMoveAssignable, long> T0;
        typedef cexa::tuple<PotentiallyThrowingMoveAssignable, int> T1;
        static_assert(std::is_assignable<T0&, T1&&>::value, "");
        static_assert(!std::is_nothrow_assignable<T0&, T1&&>::value, "");
    }
    {
        // We assign through the reference and don't move out of the incoming ref,
        // so this doesn't work (but would if the type were CopyAssignable).
        {
            using T1 = cexa::tuple<MoveAssignable&, long>;
            using T2 = cexa::tuple<MoveAssignable&, int>;
            static_assert(!std::is_assignable<T1&, T2&&>::value, "");
        }

        // ... works if it's CopyAssignable
        {
            using T1 = cexa::tuple<CopyAssignable&, long>;
            using T2 = cexa::tuple<CopyAssignable&, int>;
            static_assert(std::is_assignable<T1&, T2&&>::value, "");
        }

        // For rvalue-references, we can move-assign if the type is MoveAssignable
        // or CopyAssignable (since in the worst case the move will decay into a copy).
        {
            using T1 = cexa::tuple<MoveAssignable&&, long>;
            using T2 = cexa::tuple<MoveAssignable&&, int>;
            static_assert(std::is_assignable<T1&, T2&&>::value, "");

            using T3 = cexa::tuple<CopyAssignable&&, long>;
            using T4 = cexa::tuple<CopyAssignable&&, int>;
            static_assert(std::is_assignable<T3&, T4&&>::value, "");
        }

        // In all cases, we can't move-assign if the types are not assignable,
        // since we assign through the reference.
        {
            using T1 = cexa::tuple<NonAssignable&, long>;
            using T2 = cexa::tuple<NonAssignable&, int>;
            static_assert(!std::is_assignable<T1&, T2&&>::value, "");

            using T3 = cexa::tuple<NonAssignable&&, long>;
            using T4 = cexa::tuple<NonAssignable&&, int>;
            static_assert(!std::is_assignable<T3&, T4&&>::value, "");
        }
    }
    {
        // Make sure that we don't incorrectly move out of the source's reference.
        using Dest = cexa::tuple<TrackMove, long>;
        using Source = cexa::tuple<TrackMove&, int>;
        TrackMove track{3};
        Source src(track, 4);
        CEXA_EXPECT(!track.moved_from);

        Dest dst;
        dst = std::move(src); // here we should make a copy
        CEXA_EXPECT(!track.moved_from);
        CEXA_EXPECT_EQ(cexa::get<0>(dst).value, 3);
    }
    {
        // But we do move out of the source's reference if it's a rvalue ref
        using Dest = cexa::tuple<TrackMove, long>;
        using Source = cexa::tuple<TrackMove&&, int>;
        TrackMove track{3};
        Source src(std::move(track), 4);
        CEXA_EXPECT(!track.moved_from); // we just took a reference

        Dest dst;
        dst = std::move(src);
        CEXA_EXPECT(track.moved_from);
        CEXA_EXPECT_EQ(cexa::get<0>(dst).value, 3);
    }
    {
        // If the source holds a value, then we move out of it too
        using Dest = cexa::tuple<TrackMove, long>;
        using Source = cexa::tuple<TrackMove, int>;
        Source src(TrackMove{3}, 4);
        Dest dst;
        dst = std::move(src);
        CEXA_EXPECT(cexa::get<0>(src).moved_from);
        CEXA_EXPECT_EQ(cexa::get<0>(dst).value, 3);
    }
))
// clang-format on
