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
#include <string>

#include <gtest/gtest.h>

#include <mpark/config.hpp>

#include "util.hpp"

struct Visit_MutVarMutType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int> v(42);
    // Check `v`.
    DEXPECT_EQ(42, cexa::experimental::get<int>(v));
    // Check qualifier.
    DEXPECT_EQ(LRef, cexa::experimental::visit(get_qual{}, v));
    DEXPECT_EQ(RRef, cexa::experimental::visit(get_qual{}, std::move(v)));
  }
};

TEST(Visit, MutVarMutType) { test_helper<Visit_MutVarMutType>(); }

struct Visit_MutVarConstType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<const int> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<const int>(v));
    // Check qualifier.
    DEXPECT_EQ(ConstLRef, cexa::experimental::visit(get_qual{}, v));
    DEXPECT_EQ(ConstRRef, cexa::experimental::visit(get_qual{}, std::move(v)));
  }
};

TEST(Visit, MutVarConstType) { test_helper<Visit_MutVarConstType>(); }

struct Visit_ConstVarMutType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    const cexa::experimental::variant<int> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<int>(v));
    // Check qualifier.
    DEXPECT_EQ(ConstLRef, cexa::experimental::visit(get_qual{}, v));
    DEXPECT_EQ(ConstRRef, cexa::experimental::visit(get_qual{}, std::move(v)));

#ifdef MPARK_CPP11_CONSTEXPR
    /* constexpr */ {
      constexpr cexa::experimental::variant<int> cv(42);
      static_assert(42 == cexa::experimental::get<int>(cv), "");
      // Check qualifier.
      static_assert(ConstLRef == cexa::experimental::visit(get_qual{}, cv), "");
      static_assert(
          ConstRRef == cexa::experimental::visit(get_qual{}, std::move(cv)),
          "");
    }
#endif
  }
};

TEST(Visit, ConstVarMutType) { test_helper<Visit_ConstVarMutType>(); }

struct Visit_ConstVarConstType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    const cexa::experimental::variant<const int> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<const int>(v));
    // Check qualifier.
    DEXPECT_EQ(ConstLRef, cexa::experimental::visit(get_qual{}, v));
    DEXPECT_EQ(ConstRRef, cexa::experimental::visit(get_qual{}, std::move(v)));

#ifdef MPARK_CPP11_CONSTEXPR
    /* constexpr */ {
      constexpr cexa::experimental::variant<const int> cv(42);
      static_assert(42 == cexa::experimental::get<const int>(cv), "");
      // Check qualifier.
      static_assert(ConstLRef == cexa::experimental::visit(get_qual{}, cv), "");
      static_assert(
          ConstRRef == cexa::experimental::visit(get_qual{}, std::move(cv)),
          "");
    }
#endif
  }
};

TEST(Visit, ConstVarConstType) { test_helper<Visit_ConstVarConstType>(); }

struct concat {
  template <typename... Args>
  KOKKOS_FUNCTION test_util::DeviceString operator()(
      const Args &...args) const {
    test_util::DeviceString ret;
    std::initializer_list<int>({(ret += args, 0)...});
    return std::move(ret);
  }
};

struct Visit_Zero {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    DEXPECT_EQ("", cexa::experimental::visit(concat{}));
  }
};

TEST(Visit, Zero) { test_helper<Visit_Zero>(); }

struct Visit_Homogeneous_Double {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v("hello"),
        w("world!");
    DEXPECT_EQ("helloworld!", cexa::experimental::visit(concat{}, v, w));

#ifdef MPARK_CPP11_CONSTEXPR
    /* constexpr */ {
      constexpr cexa::experimental::variant<int, double> cv(101), cw(202),
          cx(3.3);
      struct add_ints {
        KOKKOS_FUNCTION constexpr int operator()(int lhs, int rhs) const {
          return lhs + rhs;
        }
        KOKKOS_FUNCTION constexpr int operator()(int lhs, double) const {
          return lhs;
        }
        KOKKOS_FUNCTION constexpr int operator()(double, int rhs) const {
          return rhs;
        }
        KOKKOS_FUNCTION constexpr int operator()(double, double) const {
          return 0;
        }
      };  // add
      static_assert(303 == cexa::experimental::visit(add_ints{}, cv, cw), "");
      static_assert(202 == cexa::experimental::visit(add_ints{}, cw, cx), "");
      static_assert(101 == cexa::experimental::visit(add_ints{}, cx, cv), "");
      static_assert(0 == cexa::experimental::visit(add_ints{}, cx, cx), "");
    }
#endif
  }
};

TEST(Visit_Homogeneous, Double) { test_helper<Visit_Homogeneous_Double>(); }

struct Visit_Homogeneous_Quintuple {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(101), w("+"),
        x(202), y("="), z(303);
    DEXPECT_EQ("101+202=303",
               cexa::experimental::visit(concat{}, v, w, x, y, z));
  }
};

TEST(Visit_Homogeneous, Quintuple) {
  test_helper<Visit_Homogeneous_Quintuple>();
}

struct Visit_Heterogeneous_Double {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v("hello");
    cexa::experimental::variant<double, const char *> w("world!");
    DEXPECT_EQ("helloworld!", cexa::experimental::visit(concat{}, v, w));
  }
};

TEST(Visit_Heterogeneous, Double) { test_helper<Visit_Heterogeneous_Double>(); }

struct Visit_Heterogenous_Quintuple {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, double> v(101);
    cexa::experimental::variant<const char *> w("+");
    cexa::experimental::variant<bool, test_util::DeviceString, int> x(202);
    cexa::experimental::variant<char, test_util::DeviceString, const char *> y(
        '=');
    cexa::experimental::variant<long, short> z(303L);
    DEXPECT_EQ("101+202=303",
               cexa::experimental::visit(concat{}, v, w, x, y, z));
  }
};

TEST(Visit_Heterogenous, Quintuple) {
  test_helper<Visit_Heterogenous_Quintuple>();
}

// Only there to test DeviceString, not directly relevant to variant
struct Visit_Negative_Numbers {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, double> v(-101);
    DEXPECT_EQ("-101", cexa::experimental::visit(concat{}, v));
    cexa::experimental::variant<short, double> w((short)-202);
    DEXPECT_EQ("-202", cexa::experimental::visit(concat{}, w));
    cexa::experimental::variant<long, double> x(-303L);
    DEXPECT_EQ("-303", cexa::experimental::visit(concat{}, x));
  }
};

TEST(Visit_Negative, Numbers) { test_helper<Visit_Negative_Numbers>(); }

TEST_MAIN
