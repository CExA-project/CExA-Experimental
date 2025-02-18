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

struct Ctor_Fwd_Direct {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<int>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv(42);
      static_assert(42 == cexa::experimental::get<int>(cv), "");
    }
  }
};

TEST(Ctor_Fwd, Direct) { test_helper<Ctor_Fwd_Direct>(); }

struct Ctor_Fwd_DirectConversion {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v("42");
    DEXPECT_EQ("42", cexa::experimental::get<test_util::DeviceString>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv('A');
      static_assert(65 == cexa::experimental::get<int>(cv), "");
    }
  }
};

TEST(Ctor_Fwd, DirectConversion) { test_helper<Ctor_Fwd_DirectConversion>(); }

struct Ctor_Fwd_CopyInitialization {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v = 42;
    DEXPECT_EQ(42, cexa::experimental::get<int>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv = 42;
      static_assert(42 == cexa::experimental::get<int>(cv), "");
    }
  }
};

TEST(Ctor_Fwd, CopyInitialization) {
  test_helper<Ctor_Fwd_CopyInitialization>();
}

struct Ctor_Fwd_CopyInitializationConversion {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v = "42";
    DEXPECT_EQ("42", cexa::experimental::get<test_util::DeviceString>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv = 'A';
      static_assert(65 == cexa::experimental::get<int>(cv), "");
    }
  }
};

TEST(Ctor_Fwd, CopyInitializationConversion) {
  test_helper<Ctor_Fwd_CopyInitializationConversion>();
}

TEST_MAIN
