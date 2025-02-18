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

#include <gtest/gtest.h>

#include "util.hpp"

struct Ctor_Copy_Value {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    // `v`
    cexa::experimental::variant<int, test_util::DeviceString> v("hello");
    DEXPECT_EQ("hello", cexa::experimental::get<test_util::DeviceString>(v));
    // `w`
    cexa::experimental::variant<int, test_util::DeviceString> w(v);
    DEXPECT_EQ("hello", cexa::experimental::get<test_util::DeviceString>(w));
    // Check `v`
    DEXPECT_EQ("hello", cexa::experimental::get<test_util::DeviceString>(v));

    /* constexpr */ {
      // `cv`
      constexpr cexa::experimental::variant<int, const char *> cv(42);
      static_assert(42 == cexa::experimental::get<int>(cv), "");
      // `cw`
      constexpr cexa::experimental::variant<int, const char *> cw(cv);
      static_assert(42 == cexa::experimental::get<int>(cw), "");
    }
  }
};

TEST(Ctor_Copy, Value) { test_helper<Ctor_Copy_Value>(); }

#ifdef MPARK_EXCEPTIONS
TEST(Ctor_Copy, ValuelessByException) {
  cexa::experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  cexa::experimental::variant<int, move_thrower_t> w(v);
  EXPECT_TRUE(w.valueless_by_exception());
}
#endif

TEST_MAIN
