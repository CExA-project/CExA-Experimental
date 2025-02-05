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

// HoldsAlternative<sizeof_t> only exists in variant and is not a standard
// function
TEST(Get, HoldsAlternative) {
  Cexa::Experimental::variant<int, std::string> v(42);
  // EXPECT_TRUE(Cexa::Experimental::holds_alternative<0>(v));
  // EXPECT_FALSE(Cexa::Experimental::holds_alternative<1>(v));
  EXPECT_TRUE(Cexa::Experimental::holds_alternative<int>(v));
  EXPECT_FALSE(Cexa::Experimental::holds_alternative<std::string>(v));

  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int, const char *> cv(42);
    // static_assert(Cexa::Experimental::holds_alternative<0>(cv), "");
    // static_assert(!Cexa::Experimental::holds_alternative<1>(cv), "");
    static_assert(Cexa::Experimental::holds_alternative<int>(cv), "");
    static_assert(!Cexa::Experimental::holds_alternative<const char *>(cv), "");
  }
}

TEST(Get, MutVarMutType) {
  Cexa::Experimental::variant<int> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
  // Check qualifier.
  EXPECT_EQ(LRef, get_qual(Cexa::Experimental::get<int>(v)));
  EXPECT_EQ(RRef, get_qual(Cexa::Experimental::get<int>(std::move(v))));
}

TEST(Get, MutVarConstType) {
  Cexa::Experimental::variant<const int> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<const int>(v));
  // Check qualifier.
  EXPECT_EQ(ConstLRef, get_qual(Cexa::Experimental::get<const int>(v)));
  EXPECT_EQ(ConstRRef,
            get_qual(Cexa::Experimental::get<const int>(std::move(v))));
}

TEST(Get, ConstVarMutType) {
  const Cexa::Experimental::variant<int> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
  // Check qualifier.
  EXPECT_EQ(ConstLRef, get_qual(Cexa::Experimental::get<int>(v)));
  EXPECT_EQ(ConstRRef, get_qual(Cexa::Experimental::get<int>(std::move(v))));

  /* constexpr */ {
    constexpr Cexa::Experimental::variant<int> cv(42);
    static_assert(42 == Cexa::Experimental::get<int>(cv), "");
    // Check qualifier.
    static_assert(ConstLRef == get_qual(Cexa::Experimental::get<int>(cv)), "");
    static_assert(
        ConstRRef == get_qual(Cexa::Experimental::get<int>(std::move(cv))), "");
  }
}

TEST(Get, ConstVarConstType) {
  const Cexa::Experimental::variant<const int> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<const int>(v));
  // Check qualifier.
  EXPECT_EQ(ConstLRef, get_qual(Cexa::Experimental::get<const int>(v)));
  EXPECT_EQ(ConstRRef,
            get_qual(Cexa::Experimental::get<const int>(std::move(v))));

  /* constexpr */ {
    constexpr Cexa::Experimental::variant<const int> cv(42);
    static_assert(42 == Cexa::Experimental::get<const int>(cv), "");
    // Check qualifier.
    static_assert(ConstLRef == get_qual(Cexa::Experimental::get<const int>(cv)),
                  "");
    static_assert(ConstRRef == get_qual(Cexa::Experimental::get<const int>(
                                   std::move(cv))),
                  "");
  }
}

#ifdef MPARK_EXCEPTIONS
TEST(Get, ValuelessByException) {
  Cexa::Experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  EXPECT_THROW(Cexa::Experimental::get<int>(v),
               Cexa::Experimental::bad_variant_access);
  EXPECT_THROW(Cexa::Experimental::get<move_thrower_t>(v),
               Cexa::Experimental::bad_variant_access);
}
#endif

TEST(GetIf, MutVarMutType) {
  Cexa::Experimental::variant<int> v(42);
  EXPECT_EQ(42, *Cexa::Experimental::get_if<int>(&v));
  // Check qualifier.
  EXPECT_EQ(Ptr, get_qual(Cexa::Experimental::get_if<int>(&v)));
}

TEST(GetIf, MutVarConstType) {
  Cexa::Experimental::variant<const int> v(42);
  EXPECT_EQ(42, *Cexa::Experimental::get_if<const int>(&v));
  // Check qualifier.
  EXPECT_EQ(ConstPtr, get_qual(Cexa::Experimental::get_if<const int>(&v)));
}

TEST(GetIf, ConstVarMutType) {
  const Cexa::Experimental::variant<int> v(42);
  EXPECT_EQ(42, *Cexa::Experimental::get_if<int>(&v));
  // Check qualifier.
  EXPECT_EQ(ConstPtr, get_qual(Cexa::Experimental::get_if<int>(&v)));

  /* constexpr */ {
    static constexpr Cexa::Experimental::variant<int> cv(42);
    static_assert(42 == *Cexa::Experimental::get_if<int>(&cv), "");
    // Check qualifier.
    static_assert(ConstPtr == get_qual(Cexa::Experimental::get_if<int>(&cv)),
                  "");
  }
}

TEST(GetIf, ConstVarConstType) {
  const Cexa::Experimental::variant<const int> v(42);
  EXPECT_EQ(42, *Cexa::Experimental::get_if<const int>(&v));
  // Check qualifier.
  EXPECT_EQ(ConstPtr, get_qual(Cexa::Experimental::get_if<const int>(&v)));

  /* constexpr */ {
    static constexpr Cexa::Experimental::variant<const int> cv(42);
    static_assert(42 == *Cexa::Experimental::get_if<const int>(&cv), "");
    // Check qualifier.
    static_assert(
        ConstPtr == get_qual(Cexa::Experimental::get_if<const int>(&cv)), "");
  }
}

#ifdef MPARK_EXCEPTONS
TEST(GetIf, ValuelessByException) {
  Cexa::Experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  EXPECT_EQ(nullptr, Cexa::Experimental::get_if<int>(&v));
  EXPECT_EQ(nullptr, Cexa::Experimental::get_if<move_thrower_t>(&v));
}
#endif
