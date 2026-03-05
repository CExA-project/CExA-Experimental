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
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// UNSUPPORTED: c++03

#include <type_traits>

#include <tuple.hpp>

template <class T, std::size_t N>
void test()
{
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   cexa::tuple_size<T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   cexa::tuple_size<const T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   cexa::tuple_size<volatile T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   cexa::tuple_size<const volatile T> >::value), "");
}

int main(int, char**)
{
    test<cexa::tuple<>, 0>();
    test<cexa::tuple<int>, 1>();
    test<cexa::tuple<char, int>, 2>();
    test<cexa::tuple<char, char*, int>, 3>();

  return 0;
}
