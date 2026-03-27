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
//   auto
//   operator<=>(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
// template<tuple-like UTuple>
//   friend constexpr auto operator<=>(const tuple& t, const UTuple& u); // since C++23

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <tuple.hpp>
#include <array>
#include <complex>
#include <ranges>
#include <utility>

#include <support/test_macros.h>
#include <support/cexa_test_macros.hpp>

template <class T, class U>
concept can_compare = requires(T t, U u) { t <=> u; };

using T1  = cexa::tuple<int>;
using T2  = cexa::tuple<int, long>;
using T1P = cexa::tuple<int*>;

static_assert(!can_compare<T1, T2>);
static_assert(!can_compare<T2, T1>);
static_assert(!can_compare<T1, std::array<int, 2>>);
static_assert(!can_compare<std::array<int, 2>, T1>);
static_assert(!can_compare<T1, std::pair<int, long>>);
static_assert(!can_compare<std::pair<int, long>, T1>);
static_assert(!can_compare<T1, std::complex<double>>);
static_assert(!can_compare<std::complex<double>, T1>);
static_assert(!can_compare<T1P, std::ranges::subrange<int*>>);
static_assert(!can_compare<std::ranges::subrange<int*>, T1P>);

int main() {}
