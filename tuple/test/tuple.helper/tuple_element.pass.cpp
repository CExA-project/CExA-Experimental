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

// template <size_t I, class... Types>
// struct tuple_element<I, tuple<Types...> >
// {
//     typedef Ti type;
// };

// UNSUPPORTED: c++03

#include <type_traits>

#include <tuple.hpp>

template <class T, std::size_t N, class U>
void test()
{
    static_assert((std::is_same<typename cexa::tuple_element<N, T>::type, U>::value), "");
    static_assert((std::is_same<typename cexa::tuple_element<N, const T>::type, const U>::value), "");
    static_assert((std::is_same<typename cexa::tuple_element<N, volatile T>::type, volatile U>::value), "");
    static_assert((std::is_same<typename cexa::tuple_element<N, const volatile T>::type, const volatile U>::value), "");
    static_assert((std::is_same<typename cexa::tuple_element_t<N, T>, U>::value), "");
    static_assert((std::is_same<typename cexa::tuple_element_t<N, const T>, const U>::value), "");
    static_assert((std::is_same<typename cexa::tuple_element_t<N, volatile T>, volatile U>::value), "");
    static_assert((std::is_same<typename cexa::tuple_element_t<N, const volatile T>, const volatile U>::value), "");
}

int main(int, char**)
{
    test<cexa::tuple<int>, 0, int>();
    test<cexa::tuple<char, int>, 0, char>();
    test<cexa::tuple<char, int>, 1, int>();
    test<cexa::tuple<int*, char, int>, 0, int*>();
    test<cexa::tuple<int*, char, int>, 1, char>();
    test<cexa::tuple<int*, char, int>, 2, int>();

  return 0;
}
