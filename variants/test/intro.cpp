// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <iostream>

#include <gtest/gtest.h>

#include "util.hpp"

struct Variant_Intro {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    // direct initialization.
    Cexa::Experimental::variant<int, test_util::DeviceString> v("hello world!");

    // direct access via reference.
    DEXPECT_EQ("hello world!",
               Cexa::Experimental::get<test_util::DeviceString>(v));

    // bad access.
#ifdef MPARK_EXCEPTIONS
    EXPECT_THROW(Cexa::Experimental::get<int>(v),
                 Cexa::Experimental::bad_variant_access);
#endif

    // copy construction.
    Cexa::Experimental::variant<int, test_util::DeviceString> w(v);

    // direct access via pointer.
    DEXPECT_FALSE(Cexa::Experimental::get_if<int>(&w));
    DEXPECT_TRUE(Cexa::Experimental::get_if<test_util::DeviceString>(&w));

    // diff-type assignment.
    v = 42;

    struct unary {
      KOKKOS_FUNCTION int operator()(int) const noexcept { return 0; }
      KOKKOS_FUNCTION int
      operator()(const test_util::DeviceString &) const noexcept {
        return 1;
      }
    }; // unary

    // single visitation.
    DEXPECT_EQ(0, Cexa::Experimental::visit(unary{}, v));

    // same-type assignment.
    w = "hello";

    DEXPECT_NE(v, w);

    // make `w` equal to `v`.
    w = 42;

    DEXPECT_EQ(v, w);

    struct binary {
      KOKKOS_FUNCTION int operator()(int, int) const noexcept { return 0; }
      KOKKOS_FUNCTION int
      operator()(int, const test_util::DeviceString &) const noexcept {
        return 1;
      }
      KOKKOS_FUNCTION int operator()(const test_util::DeviceString &,
                                     int) const noexcept {
        return 2;
      }
      KOKKOS_FUNCTION int
      operator()(const test_util::DeviceString &,
                 const test_util::DeviceString &) const noexcept {
        return 3;
      }
    }; // binary

    // binary visitation.
    DEXPECT_EQ(0, Cexa::Experimental::visit(binary{}, v, w));
  }
};

TEST(Variant, Intro) { test_helper<Variant_Intro>(); }

TEST_MAIN
