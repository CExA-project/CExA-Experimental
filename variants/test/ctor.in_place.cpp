// cexa::experimental.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <gtest/gtest.h>

#include "util.hpp"

struct Ctor_InPlace_IndexDirect {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(
        cexa::experimental::in_place_index_t<0>{}, 42);
    DEXPECT_EQ(42, cexa::experimental::get<0>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv(
          cexa::experimental::in_place_index_t<0>{}, 42);
      static_assert(42 == cexa::experimental::get<0>(cv), "");
    }
  }
};

TEST(Ctor_InPlace, IndexDirect) { test_helper<Ctor_InPlace_IndexDirect>(); }

struct Ctor_InPlace_IndexDirectDuplicate {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, int> v(
        cexa::experimental::in_place_index_t<0>{}, 42);
    DEXPECT_EQ(42, cexa::experimental::get<0>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, int> cv(
          cexa::experimental::in_place_index_t<0>{}, 42);
      static_assert(42 == cexa::experimental::get<0>(cv), "");
    }
  }
};

TEST(Ctor_InPlace, IndexDirectDuplicate) {
  test_helper<Ctor_InPlace_IndexDirectDuplicate>();
}

struct Ctor_InPlace_IndexConversion {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(
        cexa::experimental::in_place_index_t<1>{}, "42");
    DEXPECT_EQ("42", cexa::experimental::get<1>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv(
          cexa::experimental::in_place_index_t<0>{}, 1.1);
      static_assert(1 == cexa::experimental::get<0>(cv), "");
    }
  }
};

TEST(Ctor_InPlace, IndexConversion) {
  test_helper<Ctor_InPlace_IndexConversion>();
}

struct Ctor_InPlace_IndexInitializerList {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(
        cexa::experimental::in_place_index_t<1>{}, {'4', '2'});
    DEXPECT_EQ("42", cexa::experimental::get<1>(v));
  }
};

TEST(Ctor_InPlace, IndexInitializerList) {
  test_helper<Ctor_InPlace_IndexInitializerList>();
}

struct Ctor_InPlace_TypeDirect {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(
        cexa::experimental::in_place_type_t<test_util::DeviceString>{}, "42");
    DEXPECT_EQ("42", cexa::experimental::get<test_util::DeviceString>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv(
          cexa::experimental::in_place_type_t<int>{}, 42);
      static_assert(42 == cexa::experimental::get<int>(cv), "");
    }
  }
};

TEST(Ctor_InPlace, TypeDirect) { test_helper<Ctor_InPlace_TypeDirect>(); }

struct Ctor_InPlace_TypeConversion {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(
        cexa::experimental::in_place_type_t<int>{}, 42.5);
    DEXPECT_EQ(42, cexa::experimental::get<int>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv(
          cexa::experimental::in_place_type_t<int>{}, 42.5);
      static_assert(42 == cexa::experimental::get<int>(cv), "");
    }
  }
};

TEST(Ctor_InPlace, TypeConversion) {
  test_helper<Ctor_InPlace_TypeConversion>();
}

struct Ctor_InPlace_TypeInitializerList {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(
        cexa::experimental::in_place_type_t<test_util::DeviceString>{},
        {'4', '2'});
    DEXPECT_EQ("42", cexa::experimental::get<test_util::DeviceString>(v));
  }
};

TEST(Ctor_InPlace, TypeInitializerList) {
  test_helper<Ctor_InPlace_TypeInitializerList>();
}

TEST_MAIN
