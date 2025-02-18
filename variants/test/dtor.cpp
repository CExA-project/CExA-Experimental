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

struct Obj {
  KOKKOS_FUNCTION Obj(bool &dtor_called) : dtor_called_(dtor_called) {}
  KOKKOS_FUNCTION ~Obj() { dtor_called_ = true; }
  bool &dtor_called_;
};  // Obj

struct Dtor_Value {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    bool dtor_called = false;
    // Construct/Destruct `Obj`.
    {
      cexa::experimental::variant<Obj> v(
          cexa::experimental::in_place_type_t<Obj>{}, dtor_called);
    }
    // Check that the destructor was called.
    DEXPECT_TRUE(dtor_called);
  }
};

TEST(Dtor, Value) { test_helper<Dtor_Value>(); }

TEST_MAIN
