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
//   typename tuple_element<I, tuple<Types...> >::type&&
//   get(tuple<Types...>&& t);

// UNSUPPORTED: c++03

#include <utility>
#include <memory>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>

TEST(host_tuple_elem, get_rv_host) {
    {
        typedef cexa::tuple<std::unique_ptr<int> > T;
        T t(std::unique_ptr<int>(new int(3)));
        std::unique_ptr<int> p = cexa::get<0>(std::move(t));
        CEXA_EXPECT_EQ(*p, 3);
    }
}
