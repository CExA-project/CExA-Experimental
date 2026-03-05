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

// explicit tuple(const T&...);

// UNSUPPORTED: c++03

#include <tuple.hpp>
#include <Kokkos_Macros.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

template <class...>
struct never {
  enum { value = 0 };
};

#if defined(CEXA_ON_DEVICE)
__device__ int count = 0;
#else
int count = 0;
#endif

struct NoValueCtor {
  KOKKOS_INLINE_FUNCTION NoValueCtor() : id(++count) {}
  KOKKOS_INLINE_FUNCTION NoValueCtor(NoValueCtor const& other) : id(other.id) {
    ++count;
  }

  // The constexpr is required to make is_constructible instantiate this
  // template. The explicit is needed to test-around a similar bug with
  // is_convertible.
  template <class T>
  KOKKOS_INLINE_FUNCTION constexpr explicit NoValueCtor(T) {
    static_assert(never<T>::value, "This should not be instantiated");
  }

  int id;
};

struct NoValueCtorEmpty {
  KOKKOS_INLINE_FUNCTION NoValueCtorEmpty() {}
  KOKKOS_INLINE_FUNCTION NoValueCtorEmpty(NoValueCtorEmpty const&) {}

  template <class T>
  KOKKOS_INLINE_FUNCTION constexpr explicit NoValueCtorEmpty(T) {
    static_assert(never<T>::value, "This should not be instantiated");
  }
};

struct ImplicitCopy {
  KOKKOS_INLINE_FUNCTION explicit ImplicitCopy(int) {}
  KOKKOS_INLINE_FUNCTION ImplicitCopy(ImplicitCopy const&) {}
};

KOKKOS_INLINE_FUNCTION cexa::tuple<ImplicitCopy> testImplicitCopy1() {
  ImplicitCopy i(42);
  return {i};
}

KOKKOS_INLINE_FUNCTION cexa::tuple<ImplicitCopy> testImplicitCopy2() {
  const ImplicitCopy i(42);
  return {i};
}

KOKKOS_INLINE_FUNCTION cexa::tuple<ImplicitCopy> testImplicitCopy3() {
  const ImplicitCopy i(42);
  return i;
}

namespace test1 {
constexpr cexa::tuple<int> t(2);
static_assert(cexa::get<0>(t) == 2, "");
}  // namespace test1

namespace test2 {
constexpr cexa::tuple<int> t;
static_assert(cexa::get<0>(t) == 0, "");
}  // namespace test2

namespace test3 {
constexpr cexa::tuple<int, char*> t(2, nullptr);
static_assert(cexa::get<0>(t) == 2, "");
static_assert(cexa::get<1>(t) == nullptr, "");
}  // namespace test3

// clang-format off
CEXA_TEST(tuple_cnstr, const_Types, (
    {
        // check that the literal '0' can implicitly initialize a stored pointer.
        cexa::tuple<int*> t = 0;
        CEXA_EXPECT_EQ(cexa::get<0>(t), nullptr);
    }
    {
        cexa::tuple<int> t(2);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
    }
    {
        cexa::tuple<int, char*> t(2, 0);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t), nullptr);
    }
    {
        cexa::tuple<int, char*> t(2, nullptr);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t), nullptr);
    }
    {
        cexa::tuple<int, char*, Kokkos::pair<int, float>> t(2, nullptr, {3, 1.5f});
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t), nullptr);
        CEXA_EXPECT_EQ(cexa::get<2>(t), (Kokkos::pair{3, 1.5f}));
    }
    // __tuple_leaf<T> uses is_constructible<T, U> to disable its explicit converting
    // constructor overload __tuple_leaf(U &&). Evaluating is_constructible can cause a compile error.
    // This overload is evaluated when __tuple_leafs copy or move ctor is called.
    // This checks that is_constructible is not evaluated when U == __tuple_leaf.
    {
        cexa::tuple<int, NoValueCtor, int, int> t(1, NoValueCtor(), 2, 3);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
        CEXA_EXPECT_EQ(cexa::get<1>(t).id, 1);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<3>(t), 3);
    }
    {
        cexa::tuple<int, NoValueCtorEmpty, int, int> t(1, NoValueCtorEmpty(), 2, 3);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<3>(t), 3);
    }
))
// clang-format on
