// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)
// SPDX-FileCopyrightText: Michael Park
// SPDX-License-Identifier: BSL-1.0

#include <Kokkos_Variant.hpp>

#include <utility>

#include <gtest/gtest.h>

#include "util.hpp"

struct FunctionCalled {
  bool move_constructor_called;
  bool operator_equal_called;
};

struct Obj {
  FunctionCalled &_f;
  KOKKOS_FUNCTION constexpr Obj(FunctionCalled &f) : _f(f) {}
  KOKKOS_FUNCTION Obj(const Obj &) = delete;
  KOKKOS_FUNCTION Obj(Obj &&o) noexcept : _f(o._f) {
    _f.move_constructor_called = true;
  }
  KOKKOS_FUNCTION Obj &operator=(const Obj &) = delete;
  KOKKOS_FUNCTION Obj &operator=(Obj &&) noexcept {
    _f.operator_equal_called = true;
    return *this;
  }
};

struct Assign_Move_SameType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    // `v`, `w`.
    FunctionCalled f{false, false};
    cexa::experimental::variant<Obj, int> v(f), w(f);
    // move assignment.
    v = std::move(w);
    DEXPECT_FALSE(f.move_constructor_called);
    DEXPECT_TRUE(f.operator_equal_called);
  }
};

TEST(Assign_Move, SameType) { test_helper<Assign_Move_SameType>(); }

struct Assign_Move_DiffType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    // `v`, `w`.
    FunctionCalled f{false, false};
    cexa::experimental::variant<Obj, int> v(42), w(f);
    // move assignment.
    v = std::move(w);
    DEXPECT_TRUE(f.move_constructor_called);
    DEXPECT_FALSE(f.operator_equal_called);
  }
};

TEST(Assign_Move, DiffType) { test_helper<Assign_Move_DiffType>(); }

#ifdef EXCEPTIONS_AVAILABLE
TEST(Assign_Move, ValuelessByException) {
  cexa::experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  cexa::experimental::variant<int, move_thrower_t> w(42);
  w = std::move(v);
  EXPECT_TRUE(w.valueless_by_exception());
}
#endif

TEST_MAIN
