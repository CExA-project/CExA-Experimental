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

// template <class T> constexpr size_t tuple_size_v = tuple_size<T>::value;

#include <utility>
#include <array>

#include <tuple.hpp>

template <class Tuple, int Expect>
void test()
{
    static_assert(cexa::tuple_size_v<Tuple> == Expect, "");
    static_assert(cexa::tuple_size_v<Tuple> == cexa::tuple_size<Tuple>::value, "");
    static_assert(cexa::tuple_size_v<Tuple const> == cexa::tuple_size<Tuple>::value, "");
    static_assert(cexa::tuple_size_v<Tuple volatile> == cexa::tuple_size<Tuple>::value, "");
    static_assert(cexa::tuple_size_v<Tuple const volatile> == cexa::tuple_size<Tuple>::value, "");
}

int main(int, char**)
{
    test<cexa::tuple<>, 0>();

    test<cexa::tuple<int>, 1>();
    test<std::array<int, 1>, 1>();

    test<cexa::tuple<int, int>, 2>();
    test<std::pair<int, int>, 2>();
    test<std::array<int, 2>, 2>();

    test<cexa::tuple<int, int, int>, 3>();
    test<std::array<int, 3>, 3>();

  return 0;
}
