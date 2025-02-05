// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <mpark/config.hpp>

#include "util.hpp"

TEST(Visit, MutVarMutType) {
  Cexa::Experimental::variant<int> v(42);
  // Check `v`.
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
  // Check qualifier.
  EXPECT_EQ(LRef, Cexa::Experimental::visit(get_qual, v));
  EXPECT_EQ(RRef, Cexa::Experimental::visit(get_qual, std::move(v)));
}

TEST(Visit, MutVarConstType) {
  Cexa::Experimental::variant<const int> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<const int>(v));
  // Check qualifier.
  EXPECT_EQ(ConstLRef, Cexa::Experimental::visit(get_qual, v));
  EXPECT_EQ(ConstRRef, Cexa::Experimental::visit(get_qual, std::move(v)));
}

TEST(Visit, ConstVarMutType) {
  const Cexa::Experimental::variant<int> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
  // Check qualifier.
  EXPECT_EQ(ConstLRef, Cexa::Experimental::visit(get_qual, v));
  EXPECT_EQ(ConstRRef, Cexa::Experimental::visit(get_qual, std::move(v)));

#ifdef MPARK_CPP11_CONSTEXPR
  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int> cv(42);
    static_assert(42 == Cexa::Experimental::get<int>(cv), "");
    // Check qualifier.
    static_assert(ConstLRef == Cexa::Experimental::visit(get_qual, cv), "");
    static_assert(
        ConstRRef == Cexa::Experimental::visit(get_qual, std::move(cv)), "");
  }
#endif
}

TEST(Visit, ConstVarConstType) {
  const Cexa::Experimental::variant<const int> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<const int>(v));
  // Check qualifier.
  EXPECT_EQ(ConstLRef, Cexa::Experimental::visit(get_qual, v));
  EXPECT_EQ(ConstRRef, Cexa::Experimental::visit(get_qual, std::move(v)));

#ifdef MPARK_CPP11_CONSTEXPR
  /* constexpr */ {
    constexpr Cexa::Experimental::variant<const int> cv(42);
    static_assert(42 == Cexa::Experimental::get<const int>(cv), "");
    // Check qualifier.
    static_assert(ConstLRef == Cexa::Experimental::visit(get_qual, cv), "");
    static_assert(
        ConstRRef == Cexa::Experimental::visit(get_qual, std::move(cv)), "");
  }
#endif
}

struct concat {
  template <typename... Args>
  std::string operator()(const Args &...args) const {
    std::ostringstream strm;
    std::initializer_list<int>({(strm << args, 0)...});
    return std::move(strm).str();
  }
};

TEST(Visit, Zero) { EXPECT_EQ("", Cexa::Experimental::visit(concat{})); }

TEST(Visit_Homogeneous, Double) {
  Cexa::Experimental::variant<int, std::string> v("hello"), w("world!");
  EXPECT_EQ("helloworld!", Cexa::Experimental::visit(concat{}, v, w));

#ifdef MPARK_CPP11_CONSTEXPR
  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int, double> cv(101), cw(202),
        cx(3.3);
    struct add_ints {
      constexpr int operator()(int lhs, int rhs) const { return lhs + rhs; }
      constexpr int operator()(int lhs, double) const { return lhs; }
      constexpr int operator()(double, int rhs) const { return rhs; }
      constexpr int operator()(double, double) const { return 0; }
    }; // add
    static_assert(303 == Cexa::Experimental::visit(add_ints{}, cv, cw), "");
    static_assert(202 == Cexa::Experimental::visit(add_ints{}, cw, cx), "");
    static_assert(101 == Cexa::Experimental::visit(add_ints{}, cx, cv), "");
    static_assert(0 == Cexa::Experimental::visit(add_ints{}, cx, cx), "");
  }
#endif
}

TEST(Visit_Homogeneous, Quintuple) {
  Cexa::Experimental::variant<int, std::string> v(101), w("+"), x(202), y("="),
      z(303);
  EXPECT_EQ("101+202=303", Cexa::Experimental::visit(concat{}, v, w, x, y, z));
}

TEST(Visit_Heterogeneous, Double) {
  Cexa::Experimental::variant<int, std::string> v("hello");
  Cexa::Experimental::variant<double, const char *> w("world!");
  EXPECT_EQ("helloworld!", Cexa::Experimental::visit(concat{}, v, w));
}

TEST(Visit_Heterogenous, Quintuple) {
  Cexa::Experimental::variant<int, double> v(101);
  Cexa::Experimental::variant<const char *> w("+");
  Cexa::Experimental::variant<bool, std::string, int> x(202);
  Cexa::Experimental::variant<char, std::string, const char *> y('=');
  Cexa::Experimental::variant<long, short> z(303L);
  EXPECT_EQ("101+202=303", Cexa::Experimental::visit(concat{}, v, w, x, y, z));
}
