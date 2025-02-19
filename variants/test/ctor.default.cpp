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

struct Ctor_Default_Variant {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v;
    DEXPECT_EQ(0, cexa::experimental::get<0>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int> cv{};
      static_assert(0 == cexa::experimental::get<0>(cv), "");
    }
  }
};

TEST(Ctor_Default, Variant) { test_helper<Ctor_Default_Variant>(); }

TEST_MAIN
