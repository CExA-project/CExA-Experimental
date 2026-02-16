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

// UNSUPPORTED: c++03, c++11, c++14

// <tuple>

// Test that the constructors offered by cexa::tuple are formulated
// so they're compatible with implicit deduction guides, or if that's not
// possible that they provide explicit guides to make it work.

#include <functional>
// #include <memory>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/archetypes.h>

// clang-format off

// Overloads
//  using A = Allocator
//  using AT = std::allocator_arg_t
// ---------------
// (1)  tuple(const Types&...) -> tuple<Types...>
// (2)  tuple(pair<T1, T2>) -> tuple<T1, T2>;
// (3)  explicit tuple(const Types&...) -> tuple<Types...>
// (4)  tuple(AT, A const&, Types const&...) -> tuple<Types...>
// (5)  explicit tuple(AT, A const&, Types const&...) -> tuple<Types...>
// (6)  tuple(AT, A, pair<T1, T2>) -> tuple<T1, T2>
// (7)  tuple(tuple const& t) -> decltype(t)
// (8)  tuple(tuple&& t) -> decltype(t)
// (9)  tuple(AT, A const&, tuple const& t) -> decltype(t)
// (10) tuple(AT, A const&, tuple&& t) -> decltype(t)
CEXA_TEST(tuple_cnstr, deduct_primary, (
  // const std::allocator<int> A;
  // const auto AT = std::allocator_arg;
  { // Testing (1)
    int x = 101;
    [[maybe_unused]] cexa::tuple t1(42);
    ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<int>);
    [[maybe_unused]] cexa::tuple t2(x, 0.0, nullptr);
    ASSERT_SAME_TYPE(decltype(t2), cexa::tuple<int, double, decltype(nullptr)>);
  }
  { // Testing (3)
    using T = ExplicitTestTypes::TestType;
    static_assert(!std::is_convertible<T const&, T>::value, "");

    [[maybe_unused]] cexa::tuple t1(T{});
    ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<T>);

    // FIXME: This CTAD fails on gcc for some reasons
  #if !defined(KOKKOS_COMPILER_GNU) || KOKKOS_COMPILER_GNU >= 1300
    const T v{};
    [[maybe_unused]] cexa::tuple t2(T{}, 101l, v);
    ASSERT_SAME_TYPE(decltype(t2), cexa::tuple<T, long, T>);
  #endif
  }
  // TODO: Add when allocator_arg constructors are supported
  // { // Testing (4)
  //   int x = 101;
  //   cexa::tuple t1(AT, A, 42);
  //   ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<int>);
  //
  //   cexa::tuple t2(AT, A, 42, 0.0, x);
  //   ASSERT_SAME_TYPE(decltype(t2), cexa::tuple<int, double, int>);
  // }
  // { // Testing (5)
  //   using T = ExplicitTestTypes::TestType;
  //   static_assert(!std::is_convertible<T const&, T>::value, "");
  //
  //   cexa::tuple t1(AT, A, T{});
  //   ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<T>);
  //
  //   const T v{};
  //   cexa::tuple t2(AT, A, T{}, 101l, v);
  //   ASSERT_SAME_TYPE(decltype(t2), cexa::tuple<T, long, T>);
  // }
  // { // Testing (6)
  //   std::pair<int, char> p1(1, 'c');
  //   cexa::tuple t1(AT, A, p1);
  //   ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<int, char>);
  //
  //   std::pair<int, cexa::tuple<char, long, void*>> p2(1, cexa::tuple<char, long, void*>('c', 3l, nullptr));
  //   cexa::tuple t2(AT, A, p2);
  //   ASSERT_SAME_TYPE(decltype(t2), cexa::tuple<int, cexa::tuple<char, long, void*>>);
  //
  //   int i = 3;
  //   std::pair<std::reference_wrapper<int>, char> p3(std::ref(i), 'c');
  //   cexa::tuple t3(AT, A, p3);
  //   ASSERT_SAME_TYPE(decltype(t3), cexa::tuple<std::reference_wrapper<int>, char>);
  //
  //   std::pair<int&, char> p4(i, 'c');
  //   cexa::tuple t4(AT, A, p4);
  //   ASSERT_SAME_TYPE(decltype(t4), cexa::tuple<int&, char>);
  //
  //   cexa::tuple t5(AT, A, std::pair<int, char>(1, 'c'));
  //   ASSERT_SAME_TYPE(decltype(t5), cexa::tuple<int, char>);
  // }
  { // Testing (7)
    using Tup = cexa::tuple<int, decltype(nullptr)>;
    const Tup t(42, nullptr);

    [[maybe_unused]] cexa::tuple t1(t);
    ASSERT_SAME_TYPE(decltype(t1), Tup);
  }
  { // Testing (8)
    using Tup = cexa::tuple<void*, unsigned, char>;
    [[maybe_unused]] cexa::tuple t1(Tup(nullptr, 42, 'a'));
    ASSERT_SAME_TYPE(decltype(t1), Tup);
  }
  // { // Testing (9)
  //   using Tup = cexa::tuple<int, decltype(nullptr)>;
  //   const Tup t(42, nullptr);
  //
  //   cexa::tuple t1(AT, A, t);
  //   ASSERT_SAME_TYPE(decltype(t1), Tup);
  // }
  // { // Testing (10)
  //   using Tup = cexa::tuple<void*, unsigned, char>;
  //   cexa::tuple t1(AT, A, Tup(nullptr, 42, 'a'));
  //   ASSERT_SAME_TYPE(decltype(t1), Tup);
  // }
))

TEST(host_tuple_cnstr, deduct_primary_host) {
  { // Testing (2)
    std::pair<int, char> p1(1, 'c');
    [[maybe_unused]] cexa::tuple t1(p1);
    ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<int, char>);

    std::pair<int, cexa::tuple<char, long, void*>> p2(1, cexa::tuple<char, long, void*>('c', 3l, nullptr));
    [[maybe_unused]] cexa::tuple t2(p2);
    ASSERT_SAME_TYPE(decltype(t2), cexa::tuple<int, cexa::tuple<char, long, void*>>);

    int i = 3;
    std::pair<std::reference_wrapper<int>, char> p3(std::ref(i), 'c');
    [[maybe_unused]] cexa::tuple t3(p3);
    ASSERT_SAME_TYPE(decltype(t3), cexa::tuple<std::reference_wrapper<int>, char>);

    std::pair<int&, char> p4(i, 'c');
    [[maybe_unused]] cexa::tuple t4(p4);
    ASSERT_SAME_TYPE(decltype(t4), cexa::tuple<int&, char>);

    [[maybe_unused]] cexa::tuple t5(std::pair<int, char>(1, 'c'));
    ASSERT_SAME_TYPE(decltype(t5), cexa::tuple<int, char>);
  }
}

// Overloads
//  using A = Allocator
//  using AT = std::allocator_arg_t
// ---------------
// (1)  tuple() -> tuple<>
// (2)  tuple(AT, A const&) -> tuple<>
// (3)  tuple(tuple const&) -> tuple<>
// (4)  tuple(tuple&&) -> tuple<>
// (5)  tuple(AT, A const&, tuple const&) -> tuple<>
// (6)  tuple(AT, A const&, tuple&&) -> tuple<>
CEXA_TEST(tuple_cnstr, deduct_empty, (
  // std::allocator<int> A;
  // const auto AT = std::allocator_arg;
  { // Testing (1)
    [[maybe_unused]] cexa::tuple t1{};
    ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<>);
  }
  // { // Testing (2)
  //   cexa::tuple t1{AT, A};
  //   ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<>);
  // }
  { // Testing (3)
    const cexa::tuple<> t{};
    [[maybe_unused]] cexa::tuple t1(t);
    ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<>);
  }
  { // Testing (4)
    [[maybe_unused]] cexa::tuple t1(cexa::tuple<>{});
    ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<>);
  }
  // { // Testing (5)
  //   const cexa::tuple<> t{};
  //   cexa::tuple t1(AT, A, t);
  //   ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<>);
  // }
  // { // Testing (6)
  //   cexa::tuple t1(AT, A, cexa::tuple<>{});
  //   ASSERT_SAME_TYPE(decltype(t1), cexa::tuple<>);
  // }
))
// clang-format on
