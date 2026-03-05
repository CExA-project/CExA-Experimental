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

// template <class U1, class U2> tuple(const pair<U1, U2>& u);

// UNSUPPORTED: c++03

#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>

namespace test1 {
typedef std::pair<long, char> P0;
typedef cexa::tuple<long long, short> T1;
constexpr P0 p0(2, 'a');
constexpr T1 t1 = p0;
static_assert(cexa::get<0>(t1) == std::get<0>(p0), "");
static_assert(cexa::get<1>(t1) == std::get<1>(p0), "");
static_assert(cexa::get<0>(t1) == 2, "");
static_assert(cexa::get<1>(t1) == short('a'), "");
}  // namespace test1

TEST(tuple_cnstr, const_pair) {
  typedef std::pair<long, char> T0;
  typedef cexa::tuple<long long, short> T1;
  T0 t0(2, 'a');
  T1 t1 = t0;
  CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
  CEXA_EXPECT_EQ(cexa::get<1>(t1), short('a'));
}

// CEXA_TEST(tuple_cnstr, const_pair_kokkos, (
//         typedef Kokkos::pair<long, char> T0;
//         typedef cexa::tuple<long long, short> T1;
//         T0 t0(2, 'a');
//         T1 t1 = t0;
//         CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
//         CEXA_EXPECT_EQ(cexa::get<1>(t1), short('a'));
// ))
