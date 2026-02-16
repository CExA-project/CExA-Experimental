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

// tuple(tuple&& u);

// UNSUPPORTED: c++03

#include <utility>
// #include <memory>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/MoveOnly.h>

struct ConstructsWithTupleLeaf
{
    KOKKOS_INLINE_FUNCTION ConstructsWithTupleLeaf() {}

    KOKKOS_INLINE_FUNCTION ConstructsWithTupleLeaf(ConstructsWithTupleLeaf const &) { CEXA_EXPECT(false); }
    KOKKOS_INLINE_FUNCTION ConstructsWithTupleLeaf(ConstructsWithTupleLeaf &&) {}

    template <class T>
    KOKKOS_INLINE_FUNCTION ConstructsWithTupleLeaf(T) {
        static_assert(!std::is_same<T, T>::value,
                      "Constructor instantiated for type other than int");
    }
};

// move_only type which triggers the empty base optimization
struct move_only_ebo {
  KOKKOS_DEFAULTED_FUNCTION move_only_ebo() = default;
  KOKKOS_DEFAULTED_FUNCTION move_only_ebo(move_only_ebo&&) = default;
};

// a move_only type which does not trigger the empty base optimization
struct move_only_large final {
  KOKKOS_INLINE_FUNCTION move_only_large() : value(42) {}
  KOKKOS_DEFAULTED_FUNCTION move_only_large(move_only_large&&) = default;
  int value;
};

namespace test_move_only_ebo {
    using Tup = cexa::tuple<move_only_ebo>;
    // using Alloc = std::allocator<int>;
    // using Tag = std::allocator_arg_t;

    // special members
    static_assert(std::is_default_constructible<Tup>::value, "");
    static_assert(std::is_move_constructible<Tup>::value, "");
    static_assert(!std::is_copy_constructible<Tup>::value, "");
    static_assert(!std::is_constructible<Tup, Tup&>::value, "");

    // args constructors
    static_assert(std::is_constructible<Tup, move_only_ebo&&>::value, "");
    static_assert(!std::is_constructible<Tup, move_only_ebo const&>::value, "");
    static_assert(!std::is_constructible<Tup, move_only_ebo&>::value, "");

    // TODO: add when allocator constructors are supported
    // uses-allocator special member constructors
    // static_assert(std::is_constructible<Tup, Tag, Alloc>::value, "");
    // static_assert(std::is_constructible<Tup, Tag, Alloc, Tup&&>::value, "");
    // static_assert(!std::is_constructible<Tup, Tag, Alloc, Tup const&>::value, "");
    // static_assert(!std::is_constructible<Tup, Tag, Alloc, Tup &>::value, "");
    //
    // uses-allocator args constructors
    // static_assert(std::is_constructible<Tup, Tag, Alloc, move_only_ebo&&>::value, "");
    // static_assert(!std::is_constructible<Tup, Tag, Alloc, move_only_ebo const&>::value, "");
    // static_assert(!std::is_constructible<Tup, Tag, Alloc, move_only_ebo &>::value, "");
}

namespace test_move_only_large {
    using Tup = cexa::tuple<move_only_large>;
    // using Alloc = std::allocator<int>;
    // using Tag = std::allocator_arg_t;

    // special members
    static_assert(std::is_default_constructible<Tup>::value, "");
    static_assert(std::is_move_constructible<Tup>::value, "");
    static_assert(!std::is_copy_constructible<Tup>::value, "");
    static_assert(!std::is_constructible<Tup, Tup&>::value, "");

    // args constructors
    static_assert(std::is_constructible<Tup, move_only_large&&>::value, "");
    static_assert(!std::is_constructible<Tup, move_only_large const&>::value, "");
    static_assert(!std::is_constructible<Tup, move_only_large&>::value, "");

    // TODO: add when allocator constructors are supported
    // uses-allocator special member constructors
    // static_assert(std::is_constructible<Tup, Tag, Alloc>::value, "");
    // static_assert(std::is_constructible<Tup, Tag, Alloc, Tup&&>::value, "");
    // static_assert(!std::is_constructible<Tup, Tag, Alloc, Tup const&>::value, "");
    // static_assert(!std::is_constructible<Tup, Tag, Alloc, Tup &>::value, "");
    //
    // uses-allocator args constructors
    // static_assert(std::is_constructible<Tup, Tag, Alloc, move_only_large&&>::value, "");
    // static_assert(!std::is_constructible<Tup, Tag, Alloc, move_only_large const&>::value, "");
    // static_assert(!std::is_constructible<Tup, Tag, Alloc, move_only_large &>::value, "");
}

// clang-format off
CEXA_TEST(tuple_cnstr, move, (
    {
        typedef cexa::tuple<> T;
        T t0;
        [[maybe_unused]] T t = std::move(t0);
        // ((void)t); // Prevent unused warning
    }
    {
        typedef cexa::tuple<MoveOnly> T;
        T t0(MoveOnly(0));
        T t = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
    }
    {
        typedef cexa::tuple<MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1));
        T t = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1);
    }
    {
        typedef cexa::tuple<MoveOnly, MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
        T t = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 2);
    }
    // A bug in tuple caused __tuple_leaf to use its explicit converting constructor
    //  as its move constructor. This tests that ConstructsWithTupleLeaf is not called
    // (w/ __tuple_leaf)
    {
        typedef cexa::tuple<ConstructsWithTupleLeaf> d_t;
        d_t d((ConstructsWithTupleLeaf()));
        d_t d2(static_cast<d_t &&>(d));
    }
))
// clang-format on
