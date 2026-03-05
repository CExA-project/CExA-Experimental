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

// template <class... Types> class tuple;

// ~tuple();

// C++17 added:
//   The destructor of tuple shall be a trivial destructor
//     if (is_trivially_destructible_v<Types> && ...) is true.

#include <string>
#include <cassert>
#include <type_traits>

#include <Kokkos_Macros.hpp>
#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct TrackDtor {
  int* count_;
  KOKKOS_INLINE_FUNCTION constexpr explicit TrackDtor(int* count) : count_(count) {}
  KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 TrackDtor(TrackDtor&& that) : count_(that.count_) {
    that.count_ = nullptr;
  }
  KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX20 ~TrackDtor() {
    if (count_) ++*count_;
  }
};
static_assert(!std::is_trivially_destructible<TrackDtor>::value, "");

static_assert(std::is_trivially_destructible<cexa::tuple<>>::value, "");
static_assert(std::is_trivially_destructible<cexa::tuple<void*>>::value, "");
static_assert(std::is_trivially_destructible<cexa::tuple<int, float>>::value,
              "");
static_assert(!std::is_trivially_destructible<cexa::tuple<std::string>>::value,
              "");
static_assert(
    !std::is_trivially_destructible<cexa::tuple<int, std::string>>::value, "");

// clang-format off
CEXA_TEST(tuple_cnstr, dtor, (
  int count = 0;
  {
    cexa::tuple<TrackDtor> tuple{TrackDtor(&count)};
    CEXA_EXPECT_EQ(count, 0);
  }
  CEXA_EXPECT_EQ(count, 1);
))
// clang-format on

#if TEST_STD_VER > 17
constexpr bool test() {
  int count = 0;
  {
    cexa::tuple<TrackDtor> tuple{TrackDtor(&count)};
    assert(count == 0);
  }
  assert(count == 1);

  return true;
}
static_assert(test());
#endif
