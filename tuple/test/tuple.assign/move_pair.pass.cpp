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
//   tuple& operator=(pair<U1, U2>&& u);

// UNSUPPORTED: c++03

#include <utility>
#include <memory>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct B
{
    int id_;

    KOKKOS_INLINE_FUNCTION explicit B(int i = 0) : id_(i) {}

    KOKKOS_INLINE_FUNCTION virtual ~B() {}
};

struct D
    : B
{
    KOKKOS_INLINE_FUNCTION explicit D(int i) : B(i) {}
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

struct NonAssignable
{
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};

struct MoveAssignable
{
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  KOKKOS_DEFAULTED_FUNCTION MoveAssignable& operator=(MoveAssignable&&) = default;
};

struct CopyAssignable
{
  KOKKOS_DEFAULTED_FUNCTION CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
};

struct NothrowMoveAssignable
{
    KOKKOS_INLINE_FUNCTION NothrowMoveAssignable& operator=(NothrowMoveAssignable&&) noexcept { return *this; }
};

struct PotentiallyThrowingMoveAssignable
{
    KOKKOS_INLINE_FUNCTION PotentiallyThrowingMoveAssignable& operator=(PotentiallyThrowingMoveAssignable&&) { return *this; }
};

TEST(host_tuple_assign, move_pair)
{
    {
        typedef std::pair<long, std::unique_ptr<D>> T0;
        typedef cexa::tuple<long long, std::unique_ptr<B>> T1;
        T0 t0(2, std::unique_ptr<D>(new D(3)));
        T1 t1;
        t1 = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1)->id_, 3);
    }
    {
        using T = cexa::tuple<int, NonAssignable>;
        using P = std::pair<int, NonAssignable>;
        static_assert(!std::is_assignable<T&, P&&>::value, "");
    }
    {
      using T = cexa::tuple<int, int, int>;
      using P = std::pair<int, int>;
      static_assert(!std::is_assignable<T&, P&&>::value, "");
    }
    {
        typedef cexa::tuple<NothrowMoveAssignable, long> Tuple;
        typedef std::pair<NothrowMoveAssignable, int> Pair;
        static_assert(std::is_nothrow_assignable<Tuple&, Pair&&>::value, "");
        static_assert(!std::is_assignable<Tuple&, Pair const&&>::value, "");
    }
    {
        typedef cexa::tuple<PotentiallyThrowingMoveAssignable, long> Tuple;
        typedef std::pair<PotentiallyThrowingMoveAssignable, int> Pair;
        static_assert(std::is_assignable<Tuple&, Pair&&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Pair&&>::value, "");
        static_assert(!std::is_assignable<Tuple&, Pair const&&>::value, "");
    }
    {
        // We assign through the reference and don't move out of the incoming ref,
        // so this doesn't work (but would if the type were CopyAssignable).
        {
            using T = cexa::tuple<MoveAssignable&, int>;
            using P = std::pair<MoveAssignable&, int>;
            static_assert(!std::is_assignable<T&, P&&>::value, "");
        }

        // ... works if it's CopyAssignable
        {
            using T = cexa::tuple<CopyAssignable&, int>;
            using P = std::pair<CopyAssignable&, int>;
            static_assert(std::is_assignable<T&, P&&>::value, "");
        }

        // For rvalue-references, we can move-assign if the type is MoveAssignable
        // or CopyAssignable (since in the worst case the move will decay into a copy).
        {
            using T1 = cexa::tuple<MoveAssignable&&, int>;
            using P1 = std::pair<MoveAssignable&&, int>;
            static_assert(std::is_assignable<T1&, P1&&>::value, "");

            using T2 = cexa::tuple<CopyAssignable&&, int>;
            using P2 = std::pair<CopyAssignable&&, int>;
            static_assert(std::is_assignable<T2&, P2&&>::value, "");
        }

        // In all cases, we can't move-assign if the types are not assignable,
        // since we assign through the reference.
        {
            using T1 = cexa::tuple<NonAssignable&, int>;
            using P1 = std::pair<NonAssignable&, int>;
            static_assert(!std::is_assignable<T1&, P1&&>::value, "");

            using T2 = cexa::tuple<NonAssignable&&, int>;
            using P2 = std::pair<NonAssignable&&, int>;
            static_assert(!std::is_assignable<T2&, P2&&>::value, "");
        }
    }
    {
        // Make sure that we don't incorrectly move out of the source's reference.
        using Dest = cexa::tuple<TrackMove, int>;
        using Source = std::pair<TrackMove&, int>;
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
        using Dest = cexa::tuple<TrackMove, int>;
        using Source = std::pair<TrackMove&&, int>;
        TrackMove track{3};
        Source src(std::move(track), 4);
        CEXA_EXPECT(!track.moved_from); // we just took a reference

        Dest dst;
        dst = std::move(src);
        CEXA_EXPECT(track.moved_from);
        CEXA_EXPECT_EQ(cexa::get<0>(dst).value, 3);
    }
    {
        // If the pair holds a value, then we move out of it too
        using Dest = cexa::tuple<TrackMove, int>;
        using Source = std::pair<TrackMove, int>;
        Source src(TrackMove{3}, 4);
        Dest dst;
        dst = std::move(src);
        CEXA_EXPECT(src.first.moved_from);
        CEXA_EXPECT_EQ(cexa::get<0>(dst).value, 3);
    }
}
