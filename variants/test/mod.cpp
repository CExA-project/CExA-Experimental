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

TEST(Assign_Emplace, IndexDirect) {
  Cexa::Experimental::variant<int, std::string> v;
  v.emplace<1>("42");
  EXPECT_EQ("42", Cexa::Experimental::get<1>(v));
}

TEST(Assign_Emplace, IndexDirectDuplicate) {
  Cexa::Experimental::variant<int, int> v;
  v.emplace<1>(42);
  EXPECT_EQ(42, Cexa::Experimental::get<1>(v));
}

TEST(Assign_Emplace, IndexConversion) {
  Cexa::Experimental::variant<int, std::string> v;
  v.emplace<1>("42");
  EXPECT_EQ("42", Cexa::Experimental::get<1>(v));
}

TEST(Assign_Emplace, IndexConversionDuplicate) {
  Cexa::Experimental::variant<int, int> v;
  v.emplace<1>(1.1);
  EXPECT_EQ(1, Cexa::Experimental::get<1>(v));
}

TEST(Assign_Emplace, IndexInitializerList) {
  Cexa::Experimental::variant<int, std::string> v;
  v.emplace<1>({'4', '2'});
  EXPECT_EQ("42", Cexa::Experimental::get<1>(v));
}

TEST(Assign_Emplace, TypeDirect) {
  Cexa::Experimental::variant<int, std::string> v;
  v.emplace<std::string>("42");
  EXPECT_EQ("42", Cexa::Experimental::get<std::string>(v));
}

TEST(Assign_Emplace, TypeConversion) {
  Cexa::Experimental::variant<int, std::string> v;
  v.emplace<int>(1.1);
  EXPECT_EQ(1, Cexa::Experimental::get<int>(v));
}

TEST(Assign_Emplace, TypeInitializerList) {
  Cexa::Experimental::variant<int, std::string> v;
  v.emplace<std::string>({'4', '2'});
  EXPECT_EQ("42", Cexa::Experimental::get<std::string>(v));
}
