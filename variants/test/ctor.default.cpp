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

TEST(Ctor_Default, Variant) {
  Cexa::Experimental::variant<int, std::string> v;
  EXPECT_EQ(0, Cexa::Experimental::get<0>(v));

  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int> cv{};
    static_assert(0 == Cexa::Experimental::get<0>(cv), "");
  }
}
