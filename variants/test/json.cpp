// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

struct JsonIsh {
  JsonIsh(bool b) : data(b) {}
  JsonIsh(int i) : data(i) {}
  JsonIsh(std::string s) : data(std::move(s)) {}
  JsonIsh(std::vector<JsonIsh> v) : data(std::move(v)) {}

  Cexa::Experimental::variant<bool, int, std::string, std::vector<JsonIsh>>
      data;
};

TEST(Variant, Bool) {
  JsonIsh json_ish = true;
  EXPECT_TRUE(Cexa::Experimental::get<bool>(json_ish.data));
  json_ish = false;
  EXPECT_FALSE(Cexa::Experimental::get<bool>(json_ish.data));
}

TEST(Variant, Int) {
  JsonIsh json_ish = 42;
  EXPECT_EQ(42, Cexa::Experimental::get<int>(json_ish.data));
}

TEST(Variant, String) {
  JsonIsh json_ish = std::string("hello");
  EXPECT_EQ("hello", Cexa::Experimental::get<std::string>(json_ish.data));
}

TEST(Variant, Array) {
  JsonIsh json_ish = std::vector<JsonIsh>{true, 42, std::string("world")};
  const auto &array =
      Cexa::Experimental::get<std::vector<JsonIsh>>(json_ish.data);
  EXPECT_TRUE(Cexa::Experimental::get<bool>(array[0].data));
  EXPECT_EQ(42, Cexa::Experimental::get<int>(array[1].data));
  EXPECT_EQ("world", Cexa::Experimental::get<std::string>(array[2].data));
}
