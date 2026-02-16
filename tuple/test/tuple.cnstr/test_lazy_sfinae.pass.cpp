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

// UNSUPPORTED: c++03

// Test the following constructors:
// (1) tuple(Types const&...)
// (2) tuple(UTypes&&...)
// Test that (1) short circuits before evaluating the copy constructor of the
// second argument. Constructor (2) should be selected.

#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct NonConstCopyable {
  KOKKOS_DEFAULTED_FUNCTION NonConstCopyable() = default;
  KOKKOS_INLINE_FUNCTION explicit NonConstCopyable(int v) : value(v) {}
  KOKKOS_DEFAULTED_FUNCTION NonConstCopyable(NonConstCopyable&) = default;
  NonConstCopyable(NonConstCopyable const&) = delete;
  int value;
};

template <class T>
struct BlowsUpOnConstCopy {
  KOKKOS_DEFAULTED_FUNCTION BlowsUpOnConstCopy() = default;
  KOKKOS_INLINE_FUNCTION constexpr BlowsUpOnConstCopy(BlowsUpOnConstCopy const&) {
      static_assert(!std::is_same<T, T>::value, "");
  }
  KOKKOS_DEFAULTED_FUNCTION BlowsUpOnConstCopy(BlowsUpOnConstCopy&) = default;
};

// clang-format off
CEXA_TEST(tuple_cnstr, test_lazy_sfinae, (
  NonConstCopyable v(42);
  BlowsUpOnConstCopy<int> b;
  cexa::tuple<NonConstCopyable, BlowsUpOnConstCopy<int>> t(v, b);
  CEXA_EXPECT_EQ(cexa::get<0>(t).value, 42);
))
// clang-format on
