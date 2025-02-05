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

TEST(Ctor_Fwd, Direct) {
  Cexa::Experimental::variant<int, std::string> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));

  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int, const char *> cv(42);
    static_assert(42 == Cexa::Experimental::get<int>(cv), "");
  }
}

TEST(Ctor_Fwd, DirectConversion) {
  Cexa::Experimental::variant<int, std::string> v("42");
  EXPECT_EQ("42", Cexa::Experimental::get<std::string>(v));

  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int, const char *> cv('A');
    static_assert(65 == Cexa::Experimental::get<int>(cv), "");
  }
}

TEST(Ctor_Fwd, CopyInitialization) {
  Cexa::Experimental::variant<int, std::string> v = 42;
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));

  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int, const char *> cv = 42;
    static_assert(42 == Cexa::Experimental::get<int>(cv), "");
  }
}

TEST(Ctor_Fwd, CopyInitializationConversion) {
  Cexa::Experimental::variant<int, std::string> v = "42";
  EXPECT_EQ("42", Cexa::Experimental::get<std::string>(v));

  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int, const char *> cv = 'A';
    static_assert(65 == Cexa::Experimental::get<int>(cv), "");
  }
}
