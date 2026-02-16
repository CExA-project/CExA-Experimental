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

// UNSUPPORTED: c++03, c++11

// Make sure that we don't blow up the template instantiation recursion depth
// for tuples of size <= 512.

#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/cexa_integral_constant.hpp>
#include <support/test_macros.h>

template <std::size_t... I>
KOKKOS_INLINE_FUNCTION constexpr void CreateTuple(std::index_sequence<I...>) {
  using LargeTuple = cexa::tuple<cexa::testing::integral_constant<std::size_t, I>...>;
  using TargetTuple = cexa::tuple<decltype(I)...>;
  LargeTuple tuple(cexa::testing::integral_constant<std::size_t, I>{}...);
  auto e1 = cexa::get<0>(tuple);
  CEXA_EXPECT_EQ(e1.value, 0);
  auto e2 = cexa::get<sizeof...(I) - 1>(tuple);
  CEXA_EXPECT_EQ(e2.value, sizeof...(I) - 1);

  TargetTuple t1 = tuple;                                  // converting copy constructor from &
  TargetTuple t2 = static_cast<LargeTuple const&>(tuple);  // converting copy constructor from const&
  TargetTuple t3 = std::move(tuple);                       // converting rvalue constructor
  TargetTuple t4 = static_cast<LargeTuple const&&>(tuple); // converting const rvalue constructor
  TargetTuple t5;                                          // default constructor
  (void)t1; (void)t2; (void)t3; (void)t4; (void)t5;

#if TEST_STD_VER >= 20
  t1 = tuple;                                              // converting assignment from &
  t1 = static_cast<LargeTuple const&>(tuple);              // converting assignment from const&
  t1 = std::move(tuple);                                   // converting assignment from &&
  t1 = static_cast<LargeTuple const&&>(tuple);             // converting assignment from const&&
  swap(t1, t2);                                            // swap
#endif
  // t1 == tuple;                                          // comparison does not work yet (we blow the constexpr stack)
}

// FIXME: use less template recursion in the implementation
KOKKOS_INLINE_FUNCTION constexpr bool test() {
  // CreateTuple(std::make_index_sequence<512>{});
#if defined(KOKKOS_COMPILER_NVCC)
  CreateTuple(std::make_index_sequence<64>{});
#else
  CreateTuple(std::make_index_sequence<128>{});
#endif
  return true;
}

// clang-format off
CEXA_TEST(tuple_cnstr, recursion_depth, (
  test();
  static_assert(test(), "");
))
// clang-format off
