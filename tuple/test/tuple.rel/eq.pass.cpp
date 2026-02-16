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

// template<class... TTypes, class... UTypes>
//   bool
//   operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
// template<tuple-like UTuple>
//   friend constexpr bool operator==(const tuple& t, const UTuple& u); // since C++23

// UNSUPPORTED: c++03

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

KOKKOS_INLINE_FUNCTION constexpr bool test() {
  {
    typedef cexa::tuple<> T1;
    typedef cexa::tuple<> T2;
    const T1 t1;
    const T2 t2;
    CEXA_EXPECT(t1 == t2);
    CEXA_EXPECT(!(t1 != t2));
  }
  {
    typedef cexa::tuple<int> T1;
    typedef cexa::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(1.1);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<int> T1;
    typedef cexa::tuple<double> T2;
    const T1 t1(1);
    const T2 t2(1);
    CEXA_EXPECT(t1 == t2);
    CEXA_EXPECT(!(t1 != t2));
  }
  {
    typedef cexa::tuple<int, double> T1;
    typedef cexa::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 2);
    CEXA_EXPECT(t1 == t2);
    CEXA_EXPECT(!(t1 != t2));
  }
  {
    typedef cexa::tuple<int, double> T1;
    typedef cexa::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1, 3);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<int, double> T1;
    typedef cexa::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1.1, 2);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<int, double> T1;
    typedef cexa::tuple<double, long> T2;
    const T1 t1(1, 2);
    const T2 t2(1.1, 3);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<long, int, double> T1;
    typedef cexa::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 3);
    CEXA_EXPECT(t1 == t2);
    CEXA_EXPECT(!(t1 != t2));
  }
  {
    typedef cexa::tuple<long, int, double> T1;
    typedef cexa::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 2, 3);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<long, int, double> T1;
    typedef cexa::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 3, 3);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<long, int, double> T1;
    typedef cexa::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 2, 4);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<long, int, double> T1;
    typedef cexa::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1, 3, 2);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<long, int, double> T1;
    typedef cexa::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 2, 2);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<long, int, double> T1;
    typedef cexa::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 3, 3);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    typedef cexa::tuple<long, int, double> T1;
    typedef cexa::tuple<double, long, int> T2;
    const T1 t1(1, 2, 3);
    const T2 t2(1.1, 3, 2);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }
  {
    using T1 = cexa::tuple<long, int, double>;
    using T2 = cexa::tuple<double, long, int>;
    constexpr T1 t1(1, 2, 3);
    constexpr T2 t2(1.1, 3, 2);
    CEXA_EXPECT(!(t1 == t2));
    CEXA_EXPECT(t1 != t2);
  }

  return true;
}

// clang-format off
CEXA_TEST(tuple_rel, eq, (
  test();
  static_assert(test(), "");
))
// clang-format on
