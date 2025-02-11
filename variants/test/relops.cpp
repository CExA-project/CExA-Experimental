// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <gtest/gtest.h>

#include <mpark/config.hpp>

#include "util.hpp"

struct Rel_SameTypeSameValue {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, const char *> v(0), w(0);
    // `v` op `w`
    DEXPECT_TRUE(v == w);
    DEXPECT_FALSE(v != w);
    DEXPECT_FALSE(v < w);
    DEXPECT_FALSE(v > w);
    DEXPECT_TRUE(v <= w);
    DEXPECT_TRUE(v >= w);
    // `w` op `v`
    DEXPECT_TRUE(w == v);
    DEXPECT_FALSE(w != v);
    DEXPECT_FALSE(w < v);
    DEXPECT_FALSE(w > v);
    DEXPECT_TRUE(w <= v);
    DEXPECT_TRUE(w >= v);

#ifdef MPARK_CPP11_CONSTEXPR
    /* constexpr */ {
      constexpr Cexa::Experimental::variant<int, const char *> cv(0), cw(0);
      // `cv` op `cw`
      static_assert(cv == cw, "");
      static_assert(!(cv != cw), "");
      static_assert(!(cv < cw), "");
      static_assert(!(cv > cw), "");
      static_assert(cv <= cw, "");
      static_assert(cv >= cw, "");
      // `cw` op `cv`
      static_assert(cw == cv, "");
      static_assert(!(cw != cv), "");
      static_assert(!(cw < cv), "");
      static_assert(!(cw > cv), "");
      static_assert(cw <= cv, "");
      static_assert(cw >= cv, "");
    }
#endif
  }
};

TEST(Rel, SameTypeSameValue) { test_helper<Rel_SameTypeSameValue>(); }

struct Rel_SameTypeDiffValue {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, const char *> v(0), w(1);
    // `v` op `w`
    DEXPECT_FALSE(v == w);
    DEXPECT_TRUE(v != w);
    DEXPECT_TRUE(v < w);
    DEXPECT_FALSE(v > w);
    DEXPECT_TRUE(v <= w);
    DEXPECT_FALSE(v >= w);
    // `w` op `v`
    DEXPECT_FALSE(w == v);
    DEXPECT_TRUE(w != v);
    DEXPECT_FALSE(w < v);
    DEXPECT_TRUE(w > v);
    DEXPECT_FALSE(w <= v);
    DEXPECT_TRUE(w >= v);

#ifdef MPARK_CPP11_CONSTEXPR
    /* constexpr */ {
      constexpr Cexa::Experimental::variant<int, const char *> cv(0), cw(1);
      // `cv` op `cw`
      static_assert(!(cv == cw), "");
      static_assert(cv != cw, "");
      static_assert(cv < cw, "");
      static_assert(!(cv > cw), "");
      static_assert(cv <= cw, "");
      static_assert(!(cv >= cw), "");
      // `cw` op `cv`
      static_assert(!(cw == cv), "");
      static_assert(cw != cv, "");
      static_assert(!(cw < cv), "");
      static_assert(cw > cv, "");
      static_assert(!(cw <= cv), "");
      static_assert(cw >= cv, "");
    }
#endif
  }
};

TEST(Rel, SameTypeDiffValue) { test_helper<Rel_SameTypeDiffValue>(); }

struct Rel_DiffTypeSameValue {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, unsigned int> v(0), w(0u);
    // `v` op `w`
    DEXPECT_FALSE(v == w);
    DEXPECT_TRUE(v != w);
    DEXPECT_TRUE(v < w);
    DEXPECT_FALSE(v > w);
    DEXPECT_TRUE(v <= w);
    DEXPECT_FALSE(v >= w);
    // `w` op `v`
    DEXPECT_FALSE(w == v);
    DEXPECT_TRUE(w != v);
    DEXPECT_FALSE(w < v);
    DEXPECT_TRUE(w > v);
    DEXPECT_FALSE(w <= v);
    DEXPECT_TRUE(w >= v);

#ifdef MPARK_CPP11_CONSTEXPR
    /* constexpr */ {
      constexpr Cexa::Experimental::variant<int, unsigned int> cv(0), cw(0u);
      // `cv` op `cw`
      static_assert(!(cv == cw), "");
      static_assert(cv != cw, "");
      static_assert(cv < cw, "");
      static_assert(!(cv > cw), "");
      static_assert(cv <= cw, "");
      static_assert(!(cv >= cw), "");
      // `cw` op `cv`
      static_assert(!(cw == cv), "");
      static_assert(cw != cv, "");
      static_assert(!(cw < cv), "");
      static_assert(cw > cv, "");
      static_assert(!(cw <= cv), "");
      static_assert(cw >= cv, "");
    }
#endif
  }
};

TEST(Rel, DiffTypeSameValue) { test_helper<Rel_DiffTypeSameValue>(); }

struct Rel_DiffTypeDiffValue {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, unsigned int> v(0), w(1u);
    // `v` op `w`
    DEXPECT_FALSE(v == w);
    DEXPECT_TRUE(v != w);
    DEXPECT_TRUE(v < w);
    DEXPECT_FALSE(v > w);
    DEXPECT_TRUE(v <= w);
    DEXPECT_FALSE(v >= w);
    // `w` op `v`
    DEXPECT_FALSE(w == v);
    DEXPECT_TRUE(w != v);
    DEXPECT_FALSE(w < v);
    DEXPECT_TRUE(w > v);
    DEXPECT_FALSE(w <= v);
    DEXPECT_TRUE(w >= v);

#ifdef MPARK_CPP11_CONSTEXPR
    /* constexpr */ {
      constexpr Cexa::Experimental::variant<int, unsigned int> cv(0), cw(1u);
      // `cv` op `cw`
      static_assert(!(cv == cw), "");
      static_assert(cv != cw, "");
      static_assert(cv < cw, "");
      static_assert(!(cv > cw), "");
      static_assert(cv <= cw, "");
      static_assert(!(cv >= cw), "");
      // `cw` op `cv`
      static_assert(!(cw == cv), "");
      static_assert(cw != cv, "");
      static_assert(!(cw < cv), "");
      static_assert(cw > cv, "");
      static_assert(!(cw <= cv), "");
      static_assert(cw >= cv, "");
    }
#endif
  }
};

TEST(Rel, DiffTypeDiffValue) { test_helper<Rel_DiffTypeDiffValue>(); }

#ifdef MPARK_EXCEPTIONS
TEST(Rel, OneValuelessByException) {
  // `v` normal, `w` corrupted.
  Cexa::Experimental::variant<int, move_thrower_t> v(42), w(42);
  EXPECT_THROW(w = move_thrower_t{}, MoveConstruction);
  EXPECT_FALSE(v.valueless_by_exception());
  EXPECT_TRUE(w.valueless_by_exception());
  // `v` op `w`
  EXPECT_FALSE(v == w);
  EXPECT_TRUE(v != w);
  EXPECT_FALSE(v < w);
  EXPECT_TRUE(v > w);
  EXPECT_FALSE(v <= w);
  EXPECT_TRUE(v >= w);
}

TEST(Rel, BothValuelessByException) {
  // `v`, `w` both corrupted.
  Cexa::Experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  Cexa::Experimental::variant<int, move_thrower_t> w(v);
  EXPECT_TRUE(v.valueless_by_exception());
  EXPECT_TRUE(w.valueless_by_exception());
  // `v` op `w`
  EXPECT_TRUE(v == w);
  EXPECT_FALSE(v != w);
  EXPECT_FALSE(v < w);
  EXPECT_FALSE(v > w);
  EXPECT_TRUE(v <= w);
  EXPECT_TRUE(v >= w);
}
#endif

TEST_MAIN
