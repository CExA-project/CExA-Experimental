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

// UNSUPPORTED: c++03

// <tuple>

// inline constexpr ignore-type ignore;

#include <type_traits>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

static_assert(std::is_trivially_copyable<decltype(cexa::ignore)>::value, "");
static_assert(std::is_trivially_default_constructible<decltype(cexa::ignore)>::value, "");

[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr int test_nodiscard() { return 8294; }

KOKKOS_INLINE_FUNCTION constexpr bool test() {
  { [[maybe_unused]] auto& ignore_v = cexa::ignore; }

  { // Test that cexa::ignore provides converting assignment.
    auto& res = (cexa::ignore = 42);
    static_assert(noexcept(res = (cexa::ignore = 42)), "Must be noexcept");
    CEXA_EXPECT_EQ(&res, &cexa::ignore);
  }
  { // Test bit-field binding.
    struct S {
      unsigned int bf : 3;
    };
    S s{0b010};
    auto& res = (cexa::ignore = s.bf);
    CEXA_EXPECT_EQ(&res, &cexa::ignore);
  }
  { // Test that cexa::ignore provides copy/move constructors
    auto copy                   = cexa::ignore;
    [[maybe_unused]] auto moved = std::move(copy);
  }
  { // Test that cexa::ignore provides copy/move assignment
    auto copy  = cexa::ignore;
    copy       = cexa::ignore;
    auto moved = cexa::ignore;
    moved      = std::move(copy);
  }

  { cexa::ignore = test_nodiscard(); }

  return true;
}

// clang-format off
CEXA_TEST(tuple_general, ignore, (
  test();
  static_assert(test(), "");
))
// clang-format on
