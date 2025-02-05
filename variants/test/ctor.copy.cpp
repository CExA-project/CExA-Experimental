// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <string>

#include <gtest/gtest.h>

#include "util.hpp"

TEST(Ctor_Copy, Value) {
  // `v`
  Cexa::Experimental::variant<int, std::string> v("hello");
  EXPECT_EQ("hello", Cexa::Experimental::get<std::string>(v));
  // `w`
  Cexa::Experimental::variant<int, std::string> w(v);
  EXPECT_EQ("hello", Cexa::Experimental::get<std::string>(w));
  // Check `v`
  EXPECT_EQ("hello", Cexa::Experimental::get<std::string>(v));

  /* constexpr */ {
    // `cv`
    constexpr Cexa::Experimental::variant<int, const char *> cv(42);
    static_assert(42 == Cexa::Experimental::get<int>(cv), "");
    // `cw`
    constexpr Cexa::Experimental::variant<int, const char *> cw(cv);
    static_assert(42 == Cexa::Experimental::get<int>(cw), "");
  }
}

#ifdef MPARK_EXCEPTIONS
TEST(Ctor_Copy, ValuelessByException) {
  Cexa::Experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  Cexa::Experimental::variant<int, move_thrower_t> w(v);
  EXPECT_TRUE(w.valueless_by_exception());
}
#endif
