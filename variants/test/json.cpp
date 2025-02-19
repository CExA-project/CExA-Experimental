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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "util.hpp"

// No need to test interaction between Kokkos::Variant and std::vector on GPU
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && \
    !defined(KOKKOS_ENABLE_SYCL)
struct JsonIsh {
  JsonIsh(bool b) : data(b) {}
  JsonIsh(int i) : data(i) {}
  JsonIsh(std::string s) : data(std::move(s)) {}
  JsonIsh(std::vector<JsonIsh> v) : data(std::move(v)) {}

  cexa::experimental::variant<bool, int, std::string, std::vector<JsonIsh>>
      data;
};

TEST(Variant, Bool) {
  JsonIsh json_ish = true;
  EXPECT_TRUE(cexa::experimental::get<bool>(json_ish.data));
  json_ish = false;
  EXPECT_FALSE(cexa::experimental::get<bool>(json_ish.data));
}

TEST(Variant, Int) {
  JsonIsh json_ish = 42;
  EXPECT_EQ(42, cexa::experimental::get<int>(json_ish.data));
}

TEST(Variant, String) {
  JsonIsh json_ish = std::string("hello");
  EXPECT_EQ("hello", cexa::experimental::get<std::string>(json_ish.data));
}

TEST(Variant, Array) {
  JsonIsh json_ish = std::vector<JsonIsh>{true, 42, std::string("world")};
  const auto &array =
      cexa::experimental::get<std::vector<JsonIsh>>(json_ish.data);
  EXPECT_TRUE(cexa::experimental::get<bool>(array[0].data));
  EXPECT_EQ(42, cexa::experimental::get<int>(array[1].data));
  EXPECT_EQ("world", cexa::experimental::get<std::string>(array[2].data));
}
#endif

TEST_MAIN
