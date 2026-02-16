// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
//
// This is a modified version of the tuple tests from llvm's libcxx tests,
// below is the original copyright statement
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// explicit(see-below) constexpr tuple();

// UNSUPPORTED: c++03

#include <type_traits>

#include <Kokkos_Macros.hpp>
#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/DefaultOnly.h>

struct NoDefault {
  NoDefault() = delete;
  KOKKOS_INLINE_FUNCTION explicit NoDefault(int) {}
};

struct NoExceptDefault {
  KOKKOS_DEFAULTED_FUNCTION NoExceptDefault() noexcept = default;
};

struct ThrowingDefault {
  KOKKOS_INLINE_FUNCTION ThrowingDefault() {}
};

struct IllFormedDefault {
  KOKKOS_INLINE_FUNCTION IllFormedDefault(int x) : value(x) {}
  template <bool Pred = false>
  constexpr IllFormedDefault() {
    static_assert(Pred, "The default constructor should not be instantiated");
  }
  int value;
};

// See bug #21157.
static_assert(!std::is_default_constructible<cexa::tuple<NoDefault>>(), "");
static_assert(
    !std::is_default_constructible<cexa::tuple<DefaultOnly, NoDefault>>(), "");
static_assert(!std::is_default_constructible<
                  cexa::tuple<NoDefault, DefaultOnly, NoDefault>>(),
              "");

static_assert(noexcept(cexa::tuple<NoExceptDefault>()), "");
static_assert(noexcept(cexa::tuple<NoExceptDefault, NoExceptDefault>()), "");

static_assert(!noexcept(cexa::tuple<ThrowingDefault, NoExceptDefault>()), "");
static_assert(!noexcept(cexa::tuple<NoExceptDefault, ThrowingDefault>()), "");
static_assert(!noexcept(cexa::tuple<ThrowingDefault, ThrowingDefault>()), "");

struct Base {};
struct Derived : Base {
 protected:
  Derived() = default;
};
static_assert(!std::is_default_constructible<cexa::tuple<Derived, int>>::value,
              "");

// clang-format off
CEXA_TEST(tuple_cnstr, default, (
  {
    cexa::tuple<> t; (void)t;
  }
  {
    cexa::tuple<int> t;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
  }
  {
    cexa::tuple<int, char*> t;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
    CEXA_EXPECT_EQ(cexa::get<1>(t), nullptr);
  }
  {
    cexa::tuple<int, char*, Kokkos::pair<int, int>> t;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
    CEXA_EXPECT_EQ(cexa::get<1>(t), nullptr);
    CEXA_EXPECT_EQ(cexa::get<2>(t), (Kokkos::pair{0, 0}));
  }
  {
    cexa::tuple<int, char*, Kokkos::pair<int, int>, DefaultOnly> t;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
    CEXA_EXPECT_EQ(cexa::get<1>(t), nullptr);
    CEXA_EXPECT_EQ(cexa::get<2>(t), (Kokkos::pair{0, 0}));
    CEXA_EXPECT_EQ(cexa::get<3>(t), DefaultOnly());
  }
  {
    // Check that the SFINAE on the default constructor is not evaluated when
    // it isn't needed. If the default constructor is evaluated then this test
    // should fail to compile.
    IllFormedDefault v(0);
    cexa::tuple<IllFormedDefault> t(v);
  }
  {
    constexpr cexa::tuple<> t;
    (void)t;
  }
  {
    constexpr cexa::tuple<int> t;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
  }
  {
    constexpr cexa::tuple<int, char*> t;
    CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
    CEXA_EXPECT_EQ(cexa::get<1>(t), nullptr);
  }
))
// clang-format on
