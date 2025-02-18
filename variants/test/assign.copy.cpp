// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <gtest/gtest.h>

#include "util.hpp"

struct FunctionCalled {
  bool copy_constructor_called;
  bool operator_equal_called;
};

struct Obj {
  FunctionCalled &_f;
  KOKKOS_FUNCTION constexpr Obj(FunctionCalled &f) : _f(f) {}
  KOKKOS_FUNCTION Obj(const Obj &o) noexcept : _f(o._f) {
    _f.copy_constructor_called = true;
  }
  Obj(Obj &&) = default;
  KOKKOS_FUNCTION Obj &operator=(const Obj &) noexcept {
    _f.operator_equal_called = true;
    return *this;
  }
  Obj &operator=(Obj &&) = delete;
};

struct Assign_Copy_SameType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    // `v`, `w`.
    FunctionCalled f{false, false};
    cexa::experimental::variant<Obj, int> v(f), w(f);
    // copy assignment.
    v = w;
    DEXPECT_FALSE(f.copy_constructor_called);
    DEXPECT_TRUE(f.operator_equal_called);
  }
};

TEST(Assign_Copy, SameType) { test_helper<Assign_Copy_SameType>(); }

struct Assign_Copy_DiffType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    // `v`, `w`.
    FunctionCalled f{false, false};
    cexa::experimental::variant<Obj, int> v(42), w(f);
    // copy assignment.
    v = w;
    DEXPECT_TRUE(f.copy_constructor_called);
    DEXPECT_FALSE(f.operator_equal_called);
  }
};

TEST(Assign_Copy, DiffType) { test_helper<Assign_Copy_DiffType>(); }

#ifdef MPARK_EXCEPTIONS
TEST(Assign_Copy, ValuelessByException) {
  cexa::experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  cexa::experimental::variant<int, move_thrower_t> w(42);
  w = v;
  EXPECT_TRUE(w.valueless_by_exception());
}
#endif

TEST_MAIN
