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

struct Assign_Emplace_IndexDirect {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, test_util::DeviceString> v;
    v.emplace<1>("42");
    DEXPECT_EQ("42", Cexa::Experimental::get<1>(v));
  }
};

TEST(Assign_Emplace, IndexDirect) { test_helper<Assign_Emplace_IndexDirect>(); }

struct Assign_Emplace_IndexDirectDuplicate {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, int> v;
    v.emplace<1>(42);
    DEXPECT_EQ(42, Cexa::Experimental::get<1>(v));
  }
};

TEST(Assign_Emplace, IndexDirectDuplicate) {
  test_helper<Assign_Emplace_IndexDirectDuplicate>();
}

struct Assign_Emplace_IndexConversion {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, test_util::DeviceString> v;
    v.emplace<1>("42");
    DEXPECT_EQ("42", Cexa::Experimental::get<1>(v));
  }
};

TEST(Assign_Emplace, IndexConversion) {
  test_helper<Assign_Emplace_IndexConversion>();
}

struct Assign_Emplace_IndexConversionDuplicate {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, int> v;
    v.emplace<1>(1.1);
    DEXPECT_EQ(1, Cexa::Experimental::get<1>(v));
  }
};

TEST(Assign_Emplace, IndexConversionDuplicate) {
  test_helper<Assign_Emplace_IndexConversionDuplicate>();
}

struct Assign_Emplace_IndexInitializerList {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, test_util::DeviceString> v;
    v.emplace<1>({'4', '2'});
    DEXPECT_EQ("42", Cexa::Experimental::get<1>(v));
  }
};

TEST(Assign_Emplace, IndexInitializerList) {
  test_helper<Assign_Emplace_IndexInitializerList>();
}

struct Assign_Emplace_TypeDirect {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, test_util::DeviceString> v;
    v.emplace<test_util::DeviceString>("42");
    DEXPECT_EQ("42", Cexa::Experimental::get<test_util::DeviceString>(v));
  }
};

TEST(Assign_Emplace, TypeDirect) { test_helper<Assign_Emplace_TypeDirect>(); }

struct Assign_Emplace_TypeConversion {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, test_util::DeviceString> v;
    v.emplace<int>(1.1);
    DEXPECT_EQ(1, Cexa::Experimental::get<int>(v));
  }
};

TEST(Assign_Emplace, TypeConversion) {
  test_helper<Assign_Emplace_TypeConversion>();
}

struct Assign_Emplace_TypeInitializerList {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, test_util::DeviceString> v;
    v.emplace<test_util::DeviceString>({'4', '2'});
    DEXPECT_EQ("42", Cexa::Experimental::get<test_util::DeviceString>(v));
  }
};

TEST(Assign_Emplace, TypeInitializerList) {
  test_helper<Assign_Emplace_TypeInitializerList>();
}

TEST_MAIN
