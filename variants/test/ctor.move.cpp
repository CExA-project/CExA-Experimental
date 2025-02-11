// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <utility>

#include <gtest/gtest.h>

#include "util.hpp"

struct Ctor_Move_Value {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    // `v`
    Cexa::Experimental::variant<int, test_util::DeviceString> v("hello");
    DEXPECT_EQ("hello", Cexa::Experimental::get<test_util::DeviceString>(v));
    // `w`
    Cexa::Experimental::variant<int, test_util::DeviceString> w(std::move(v));
    DEXPECT_EQ("hello", Cexa::Experimental::get<test_util::DeviceString>(w));

    /* constexpr */ {
      // `cv`
      constexpr Cexa::Experimental::variant<int, const char *> cv(42);
      static_assert(42 == Cexa::Experimental::get<int>(cv), "");
      // `cw`
      constexpr Cexa::Experimental::variant<int, const char *> cw(
          std::move(cv));
      static_assert(42 == Cexa::Experimental::get<int>(cw), "");
    }
  }
};

TEST(Ctor_Move, Value) { test_helper<Ctor_Move_Value>(); }

#ifdef MPARK_EXCEPTIONS
TEST(Ctor_Move, ValuelessByException) {
  Cexa::Experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  Cexa::Experimental::variant<int, move_thrower_t> w(std::move(v));
  EXPECT_TRUE(w.valueless_by_exception());
}
#endif

TEST_MAIN
