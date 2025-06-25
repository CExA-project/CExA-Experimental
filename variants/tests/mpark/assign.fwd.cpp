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

#include <sstream>
#include <type_traits>

#include <gtest/gtest.h>

#include "../util.hpp"

struct Assign_Fwd_SameType {
  KOKKOS_FUNCTION void operator()(const int i, int &errors) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(101);
    DEXPECT_EQ(101, cexa::experimental::get<int>(v));
    v = 202;
    DEXPECT_EQ(202, cexa::experimental::get<int>(v));
  }
};

TEST(Assign_Fwd, SameType) { test_util::test_helper<Assign_Fwd_SameType>(); }

struct Assign_Fwd_DiffType {
  KOKKOS_FUNCTION void operator()(const int i, int &errors) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<int>(v));
    v = "42";
    DEXPECT_EQ("42", cexa::experimental::get<test_util::DeviceString>(v));
  }
};

TEST(Assign_Fwd, DiffType) { test_util::test_helper<Assign_Fwd_DiffType>(); }

struct Assign_Fwd_ExactMatch {
  KOKKOS_FUNCTION void operator()(const int i, int &errors) const {
    cexa::experimental::variant<const char *, test_util::DeviceString> v;
    v = test_util::DeviceString("hello");
    DEXPECT_EQ("hello", cexa::experimental::get<test_util::DeviceString>(v));
  }
};

TEST(Assign_Fwd, ExactMatch) {
  test_util::test_helper<Assign_Fwd_ExactMatch>();
}

struct Assign_Fwd_BetterMatch {
  KOKKOS_FUNCTION void operator()(const int i, int &errors) const {
    cexa::experimental::variant<int, double> v;
    // `char` -> `int` is better than `char` -> `double`
    v = 'x';
    DEXPECT_EQ(static_cast<int>('x'), cexa::experimental::get<int>(v));
  }
};

TEST(Assign_Fwd, BetterMatch) {
  test_util::test_helper<Assign_Fwd_BetterMatch>();
}

TEST(Assign_Fwd, NoMatch) {
  struct x {};
  static_assert(
      !std::is_assignable<
          cexa::experimental::variant<int, test_util::DeviceString>, x>{},
      "variant<int, test_util::DeviceString> v; v = x;");
}

TEST(Assign_Fwd, WideningOrAmbiguous) {
  // There is a bug in the cuda implementation of variants (see here:
  // https://github.com/NVIDIA/cccl/issues/4395)
#if !defined(KOKKOS_COMPILER_NVCC) || KOKKOS_COMPILER_NVCC < 1250
  static_assert(
      std::is_assignable<cexa::experimental::variant<short, long>, int>{},
      "variant<short, long> v; v = 42;");
#endif
}

struct Assign_Fwd_SameTypeOptimization {
  KOKKOS_FUNCTION void operator()(const int i, int &errors) const {
    cexa::experimental::variant<int, test_util::DeviceString> v("hello world!");
    // Check `v`.
    const test_util::DeviceString &x =
        cexa::experimental::get<test_util::DeviceString>(v);
    DEXPECT_EQ("hello world!", x);
    // Save the "hello world!"'s capacity.
    auto capacity = x.capacity();
    // Use `test_util::DeviceString::operator=(const char *)` to assign into
    // `v`.
    v = "hello";
    // Check `v`.
    const test_util::DeviceString &y =
        cexa::experimental::get<test_util::DeviceString>(v);
    DEXPECT_EQ("hello", y);
    // Since "hello" is shorter than "hello world!", we should have preserved
    // the existing capacity of the string!.
    DEXPECT_EQ(capacity, y.capacity());
  }
};

TEST(Assign_Fwd, SameTypeOptimization) {
  test_util::test_helper<Assign_Fwd_SameTypeOptimization>();
}

#ifdef EXCEPTIONS_AVAILABLE
TEST(Assign_Fwd, ThrowOnAssignment) {
  cexa::experimental::variant<int, move_thrower_t> v(
      cexa::experimental::in_place_type_t<move_thrower_t>{});
  // Since `variant` is already in `move_thrower_t`, assignment optimization
  // kicks and we simply invoke
  // `move_thrower_t &operator=(move_thrower_t &&);` which throws.
  EXPECT_THROW(v = move_thrower_t{}, MoveAssignment);
  EXPECT_FALSE(v.valueless_by_exception());
  EXPECT_EQ(1u, v.index());
  // We can still assign into a variant in an invalid state.
  v = 42;
  // Check `v`.
  EXPECT_FALSE(v.valueless_by_exception());
  EXPECT_EQ(42, cexa::experimental::get<int>(v));
}
#endif

TEST_MAIN
