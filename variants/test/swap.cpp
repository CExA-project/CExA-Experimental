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

#include "util.hpp"

struct Swap_Same {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, test_util::DeviceString> v("hello");
    Cexa::Experimental::variant<int, test_util::DeviceString> w("world");
    // Check `v`.
    DEXPECT_EQ("hello", Cexa::Experimental::get<test_util::DeviceString>(v));
    // Check `w`.
    DEXPECT_EQ("world", Cexa::Experimental::get<test_util::DeviceString>(w));
    // Swap.
    Cexa::Experimental::swap(v, w);
    // Check `v`.
    DEXPECT_EQ("world", Cexa::Experimental::get<test_util::DeviceString>(v));
    // Check `w`.
    DEXPECT_EQ("hello", Cexa::Experimental::get<test_util::DeviceString>(w));
  }
};

TEST(Swap, Same) { test_helper<Swap_Same>(); }

struct Swap_Different {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    Cexa::Experimental::variant<int, test_util::DeviceString> v(42);
    Cexa::Experimental::variant<int, test_util::DeviceString> w("hello");
    // Check `v`.
    DEXPECT_EQ(42, Cexa::Experimental::get<int>(v));
    // Check `w`.
    DEXPECT_EQ("hello", Cexa::Experimental::get<test_util::DeviceString>(w));
    // Swap.
    Cexa::Experimental::swap(v, w);
    // Check `v`.
    DEXPECT_EQ("hello", Cexa::Experimental::get<test_util::DeviceString>(v));
    // Check `w`.
    DEXPECT_EQ(42, Cexa::Experimental::get<int>(w));
  }
};

TEST(Swap, Different) { test_helper<Swap_Different>(); }

#ifdef MPARK_EXCEPTIONS
TEST(Swap, OneValuelessByException) {
  // `v` normal, `w` corrupted.
  Cexa::Experimental::variant<int, move_thrower_t> v(42), w(42);
  EXPECT_THROW(w = move_thrower_t{}, MoveConstruction);
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
  EXPECT_TRUE(w.valueless_by_exception());
  // Swap.
  Cexa::Experimental::swap(v, w);
  // Check `v`, `w`.
  EXPECT_TRUE(v.valueless_by_exception());
  EXPECT_EQ(42, Cexa::Experimental::get<int>(w));
}

TEST(Swap, BothValuelessByException) {
  // `v`, `w` both corrupted.
  Cexa::Experimental::variant<int, move_thrower_t> v(42);
  EXPECT_THROW(v = move_thrower_t{}, MoveConstruction);
  Cexa::Experimental::variant<int, move_thrower_t> w(v);
  EXPECT_TRUE(v.valueless_by_exception());
  EXPECT_TRUE(w.valueless_by_exception());
  // Swap.
  Cexa::Experimental::swap(v, w);
  // Check `v`, `w`.
  EXPECT_TRUE(v.valueless_by_exception());
  EXPECT_TRUE(w.valueless_by_exception());
}
#endif

struct Swap_DtorsSame {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    struct Obj {
      KOKKOS_FUNCTION Obj(size_t *dtor_count) : dtor_count_(dtor_count) {}
      KOKKOS_FUNCTION Obj(const Obj &) = default;
      KOKKOS_FUNCTION Obj(Obj &&) = default;
      KOKKOS_FUNCTION ~Obj() { ++(*dtor_count_); }
      KOKKOS_FUNCTION Obj &operator=(const Obj &) = default;
      KOKKOS_FUNCTION Obj &operator=(Obj &&) = default;
      size_t *dtor_count_;
    }; // Obj
    size_t v_count = 0;
    size_t w_count = 0;
    {
      Cexa::Experimental::variant<Obj> v{&v_count}, w{&w_count};
      Cexa::Experimental::swap(v, w);
      // Calls `Cexa::Experimental::swap(Obj &lhs, Obj &rhs)`, with which we
      // perform:
      // ```
      // {
      //   Obj temp(move(lhs));
      //   lhs = move(rhs);
      //   rhs = move(temp);
      // }  `++v_count` from `temp::~Obj()`.
      // ```
      DEXPECT_EQ(1u, v_count);
      DEXPECT_EQ(0u, w_count);
    }
    DEXPECT_EQ(2u, v_count);
    DEXPECT_EQ(1u, w_count);
  }
};

TEST(Swap, DtorsSame) { test_helper<Swap_DtorsSame>(); }

namespace detail {

struct Obj {
  KOKKOS_FUNCTION Obj(size_t *dtor_count) : dtor_count_(dtor_count) {}
  KOKKOS_FUNCTION Obj(const Obj &) = default;
  KOKKOS_FUNCTION Obj(Obj &&) = default;
  KOKKOS_FUNCTION ~Obj() { ++(*dtor_count_); }
  KOKKOS_FUNCTION Obj &operator=(const Obj &) = default;
  KOKKOS_FUNCTION Obj &operator=(Obj &&) = default;
  size_t *dtor_count_;
}; // Obj

KOKKOS_FUNCTION static void swap(Obj &lhs, Obj &rhs) noexcept {
  Cexa::Experimental::swap(lhs.dtor_count_, rhs.dtor_count_);
}

} // namespace detail

struct Swap_DtorsSameWithSwap {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    size_t v_count = 0;
    size_t w_count = 0;
    {
      Cexa::Experimental::variant<detail::Obj> v{&v_count}, w{&w_count};
      using Cexa::Experimental::swap;
      swap(v, w);
      // Calls `detail::swap(Obj &lhs, Obj &rhs)`, with which doesn't call any
      // destructors.
      DEXPECT_EQ(0u, v_count);
      DEXPECT_EQ(0u, w_count);
    }
    DEXPECT_EQ(1u, v_count);
    DEXPECT_EQ(1u, w_count);
  }
};

TEST(Swap, DtorsSameWithSwap) { test_helper<Swap_DtorsSameWithSwap>(); }

struct Swap_DtorsDifferent {
  KOKKOS_FUNCTION void operator()(const int i, int &error) const {
    struct V {
      KOKKOS_FUNCTION V(size_t *dtor_count) : dtor_count_(dtor_count) {}
      KOKKOS_FUNCTION V(const V &) = default;
      KOKKOS_FUNCTION V(V &&) = default;
      KOKKOS_FUNCTION ~V() { ++(*dtor_count_); }
      KOKKOS_FUNCTION V &operator=(const V &) = default;
      KOKKOS_FUNCTION V &operator=(V &&) = default;
      size_t *dtor_count_;
    }; // V
    struct W {
      KOKKOS_FUNCTION W(size_t *dtor_count) : dtor_count_(dtor_count) {}
      KOKKOS_FUNCTION W(const W &) = default;
      KOKKOS_FUNCTION W(W &&) = default;
      KOKKOS_FUNCTION ~W() { ++(*dtor_count_); }
      KOKKOS_FUNCTION W &operator=(const W &) = default;
      KOKKOS_FUNCTION W &operator=(W &&) = default;
      size_t *dtor_count_;
    }; // W
    size_t v_count = 0;
    size_t w_count = 0;
    {
      Cexa::Experimental::variant<V, W> v{
          Cexa::Experimental::in_place_type_t<V>{}, &v_count};
      Cexa::Experimental::variant<V, W> w{
          Cexa::Experimental::in_place_type_t<W>{}, &w_count};
      using Cexa::Experimental::swap;
      swap(v, w);
      DEXPECT_EQ(1u, v_count);
      DEXPECT_EQ(2u, w_count);
    }
    DEXPECT_EQ(2u, v_count);
    DEXPECT_EQ(3u, w_count);
  }
};

TEST(Swap, DtorsDifferent) { test_helper<Swap_DtorsDifferent>(); }

TEST_MAIN
