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

// template <class... Types>
//   struct tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// UNSUPPORTED: c++03, c++11, c++14

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct S { int x; };

// TODO: maybe remove this as it is unrelated
KOKKOS_INLINE_FUNCTION void test_decomp_user_type() {
  {
    S s{99};
    auto [m1] = s;
    auto& [r1] = s;
    CEXA_EXPECT_EQ(m1, 99);
    CEXA_EXPECT_EQ(&r1, &s.x);
  }
  {
    S const s{99};
    auto [m1] = s;
    auto& [r1] = s;
    CEXA_EXPECT_EQ(m1, 99);
    CEXA_EXPECT_EQ(&r1, &s.x);
  }
}

KOKKOS_INLINE_FUNCTION void test_decomp_tuple() {
  typedef cexa::tuple<int> T;
  {
    T s{99};
    auto [m1] = s;
    auto& [r1] = s;
    CEXA_EXPECT_EQ(m1, 99);
    CEXA_EXPECT_EQ(&r1, &cexa::get<0>(s));
  }
  {
    T const s{99};
    auto [m1] = s;
    auto& [r1] = s;
    CEXA_EXPECT_EQ(m1, 99);
    CEXA_EXPECT_EQ(&r1, &cexa::get<0>(s));
  }
}


// NOTE: cexa::get is not supposed to work on std::pair
// void test_decomp_pair() {
//   typedef std::pair<int, double> T;
//   {
//     T s{99, 42.5};
//     auto [m1, m2] = s;
//     auto& [r1, r2] = s;
//     CEXA_EXPECT_EQ(m1, 99);
//     CEXA_EXPECT_EQ(m2, 42.5);
//     CEXA_EXPECT_EQ(&r1, &cexa::get<0>(s));
//     CEXA_EXPECT_EQ(&r2, &cexa::get<1>(s));
//   }
//   {
//     T const s{99, 42.5};
//     auto [m1, m2] = s;
//     auto& [r1, r2] = s;
//     CEXA_EXPECT_EQ(m1, 99);
//     CEXA_EXPECT_EQ(m2, 42.5);
//     CEXA_EXPECT_EQ(&r1, &cexa::get<0>(s));
//     CEXA_EXPECT_EQ(&r2, &cexa::get<1>(s));
//   }
// }

// NOTE: cexa::get is not supposed to work on std::array
// void test_decomp_array() {
//   typedef std::array<int, 3> T;
//   {
//     T s{{99, 42, -1}};
//     auto [m1, m2, m3] = s;
//     auto& [r1, r2, r3] = s;
//     CEXA_EXPECT_EQ(m1, 99);
//     CEXA_EXPECT_EQ(m2, 42);
//     CEXA_EXPECT_EQ(m3, -1);
//     CEXA_EXPECT_EQ(&r1, &cexa::get<0>(s));
//     CEXA_EXPECT_EQ(&r2, &cexa::get<1>(s));
//     CEXA_EXPECT_EQ(&r3, &cexa::get<2>(s));
//   }
//   {
//     T const s{{99, 42, -1}};
//     auto [m1, m2, m3] = s;
//     auto& [r1, r2, r3] = s;
//     CEXA_EXPECT_EQ(m1, 99);
//     CEXA_EXPECT_EQ(m2, 42);
//     CEXA_EXPECT_EQ(m3, -1);
//     CEXA_EXPECT_EQ(&r1, &cexa::get<0>(s));
//     CEXA_EXPECT_EQ(&r2, &cexa::get<1>(s));
//     CEXA_EXPECT_EQ(&r3, &cexa::get<2>(s));
//   }
// }

struct TestLWG2770 {
  int n;
};

template <>
struct cexa::tuple_size<TestLWG2770> {};

KOKKOS_INLINE_FUNCTION void test_lwg_2770() {
  {
    auto [n] = TestLWG2770{42};
    CEXA_EXPECT_EQ(n, 42);
  }
  {
    const auto [n] = TestLWG2770{42};
    CEXA_EXPECT_EQ(n, 42);
  }
  {
    TestLWG2770 s{42};
    auto& [n] = s;
    CEXA_EXPECT_EQ(n, 42);
    CEXA_EXPECT_EQ(&n, &s.n);
  }
  {
    const TestLWG2770 s{42};
    auto& [n] = s;
    CEXA_EXPECT_EQ(n, 42);
    CEXA_EXPECT_EQ(&n, &s.n);
  }
}

struct Test {
  int x;
};

template <std::size_t N>
KOKKOS_INLINE_FUNCTION int get(Test const&) { static_assert(N == 0, ""); return -1; }

template <>
struct cexa::tuple_element<0, Test> {
  typedef int type;
};

KOKKOS_INLINE_FUNCTION void test_before_tuple_size_specialization() {
  Test const t{99};
  auto& [p] = t;
  CEXA_EXPECT_EQ(p, 99);
}

template <>
struct cexa::tuple_size<Test> {
public:
  static const std::size_t value = 1;
};

// NOTE: Removed this as it fails on non libcxx implementations
// void test_after_tuple_size_specialization() {
//   Test const t{99};
//   auto& [p] = t;
//   // https://cplusplus.github.io/LWG/issue4040
//   // It is controversial whether cexa::tuple_size<const Test> is instantiated here or before.
//   (void)p;
//   LIBCPP_ASSERT(p == -1);
// }

// FIXME: add this later
// #if TEST_STD_VER >= 26 && __cpp_structured_bindings >= 202411L
// struct InvalidWhenNoCv1 {};
//
// template <>
// struct cexa::tuple_size<InvalidWhenNoCv1> {};
//
// struct InvalidWhenNoCv2 {};
//
// template <>
// struct cexa::tuple_size<InvalidWhenNoCv2> {
//   void value();
// };
//
// template <class = void>
// void test_decomp_as_empty_pack() {
//   {
//     const auto [... pack] = InvalidWhenNoCv1{};
//     static_assert(sizeof...(pack) == 0);
//   }
//   {
//     const auto [... pack] = InvalidWhenNoCv2{};
//     static_assert(sizeof...(pack) == 0);
//   }
// }
// #endif

CEXA_TEST(tuple_helper, tuple_size_structured_bindings, (
  test_decomp_user_type();
  test_decomp_tuple();
  // test_decomp_pair();
  // test_decomp_array();
  test_lwg_2770();
  test_before_tuple_size_specialization();
  // test_after_tuple_size_specialization();
// #if TEST_STD_VER >= 26 && __cpp_structured_bindings >= 202411L
//   test_decomp_as_empty_pack();
// #endif
))
