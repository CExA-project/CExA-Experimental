// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <utility>

#include <gtest/gtest.h>

#include "util.hpp"

struct Get_HoldsAlternative {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int, test_util::DeviceString> v(42);
    DEXPECT_TRUE(cexa::experimental::holds_alternative<int>(v));
    DEXPECT_FALSE(
        cexa::experimental::holds_alternative<test_util::DeviceString>(v));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int, const char *> cv(42);
      static_assert(cexa::experimental::holds_alternative<int>(cv), "");
      static_assert(!cexa::experimental::holds_alternative<const char *>(cv),
                    "");
    }
  }
};

TEST(Get, HoldsAlternative) { test_helper<Get_HoldsAlternative>(); }

struct Get_MutVarMutType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<int>(v));
    // Check qualifier.
    DEXPECT_EQ(LRef, get_qual{}(cexa::experimental::get<int>(v)));
    DEXPECT_EQ(RRef, get_qual{}(cexa::experimental::get<int>(std::move(v))));
  }
};

TEST(Get, MutVarMutType) { test_helper<Get_MutVarMutType>(); }

struct Get_MutVarConstType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<const int> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<const int>(v));
    // Check qualifier.
    DEXPECT_EQ(ConstLRef, get_qual{}(cexa::experimental::get<const int>(v)));
    DEXPECT_EQ(ConstRRef,
               get_qual{}(cexa::experimental::get<const int>(std::move(v))));
  }
};

TEST(Get, MutVarConstType) { test_helper<Get_MutVarConstType>(); }

struct Get_ConstVarMutType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    const cexa::experimental::variant<int> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<int>(v));
    // Check qualifier.
    DEXPECT_EQ(ConstLRef, get_qual{}(cexa::experimental::get<int>(v)));
    DEXPECT_EQ(ConstRRef,
               get_qual{}(cexa::experimental::get<int>(std::move(v))));

    /* constexpr */ {
      constexpr cexa::experimental::variant<int> cv(42);
      static_assert(42 == cexa::experimental::get<int>(cv), "");
      // Check qualifier.
      static_assert(ConstLRef == get_qual{}(cexa::experimental::get<int>(cv)),
                    "");
      static_assert(
          ConstRRef == get_qual{}(cexa::experimental::get<int>(std::move(cv))),
          "");
    }
  }
};

TEST(Get, ConstVarMutType) { test_helper<Get_ConstVarMutType>(); }

struct Get_ConstVarConstType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    const cexa::experimental::variant<const int> v(42);
    DEXPECT_EQ(42, cexa::experimental::get<const int>(v));
    // Check qualifier.
    DEXPECT_EQ(ConstLRef, get_qual{}(cexa::experimental::get<const int>(v)));
    DEXPECT_EQ(ConstRRef,
               get_qual{}(cexa::experimental::get<const int>(std::move(v))));

    /* constexpr */ {
      constexpr cexa::experimental::variant<const int> cv(42);
      static_assert(42 == cexa::experimental::get<const int>(cv), "");
      // Check qualifier.
      static_assert(
          ConstLRef == get_qual{}(cexa::experimental::get<const int>(cv)), "");
      static_assert(ConstRRef == get_qual{}(cexa::experimental::get<const int>(
                                     std::move(cv))),
                    "");
    }
  }
};

TEST(Get, ConstVarConstType) { test_helper<Get_ConstVarConstType>(); }

#ifdef MPARK_EXCEPTIONS
TEST(Get, ValuelessByException) {
  cexa::experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  EXPECT_THROW(cexa::experimental::get<int>(v),
               cexa::experimental::bad_variant_access);
  EXPECT_THROW(cexa::experimental::get<move_thrower_t>(v),
               cexa::experimental::bad_variant_access);
}
#endif

struct GetIf_MutVarMutType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<int> v(42);
    DEXPECT_EQ(42, *cexa::experimental::get_if<int>(&v));
    // Check qualifier.
    DEXPECT_EQ(Ptr, get_qual{}(cexa::experimental::get_if<int>(&v)));
  }
};

TEST(GetIf, MutVarMutType) { test_helper<GetIf_MutVarMutType>(); }

struct GetIf_MutVarConstType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    cexa::experimental::variant<const int> v(42);
    DEXPECT_EQ(42, *cexa::experimental::get_if<const int>(&v));
    // Check qualifier.
    DEXPECT_EQ(ConstPtr, get_qual{}(cexa::experimental::get_if<const int>(&v)));
  }
};

TEST(GetIf, MutVarConstType) { test_helper<GetIf_MutVarConstType>(); }

struct GetIf_ConstVarMutType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    const cexa::experimental::variant<int> v(42);
    DEXPECT_EQ(42, *cexa::experimental::get_if<int>(&v));
    // Check qualifier.
    DEXPECT_EQ(ConstPtr, get_qual{}(cexa::experimental::get_if<int>(&v)));

    /* constexpr */ {
      static constexpr cexa::experimental::variant<int> cv(42);
      static_assert(42 == *cexa::experimental::get_if<int>(&cv), "");
      // Check qualifier.
      static_assert(
          ConstPtr == get_qual{}(cexa::experimental::get_if<int>(&cv)), "");
    }
  }
};

TEST(GetIf, ConstVarMutType) { test_helper<GetIf_ConstVarMutType>(); }

struct GetIf_ConstVarConstType {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    const cexa::experimental::variant<const int> v(42);
    DEXPECT_EQ(42, *cexa::experimental::get_if<const int>(&v));
    // Check qualifier.
    DEXPECT_EQ(ConstPtr, get_qual{}(cexa::experimental::get_if<const int>(&v)));

    /* constexpr */ {
      static constexpr cexa::experimental::variant<const int> cv(42);
      static_assert(42 == *cexa::experimental::get_if<const int>(&cv), "");
      // Check qualifier.
      static_assert(
          ConstPtr == get_qual{}(cexa::experimental::get_if<const int>(&cv)),
          "");
    }
  }
};

TEST(GetIf, ConstVarConstType) { test_helper<GetIf_ConstVarConstType>(); }

#ifdef MPARK_EXCEPTONS
TEST(GetIf, ValuelessByException) {
  cexa::experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  EXPECT_EQ(nullptr, cexa::experimental::get_if<int>(&v));
  EXPECT_EQ(nullptr, cexa::experimental::get_if<move_thrower_t>(&v));
}
#endif

TEST_MAIN
