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

// template <class U1, class U2> tuple(pair<U1, U2>&& u);

// UNSUPPORTED: c++03

#include <utility>
#include <memory>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct B {
  int id_;

  KOKKOS_INLINE_FUNCTION explicit B(int i) : id_(i) {}

  KOKKOS_INLINE_FUNCTION virtual ~B() {}
};

struct D : B {
  KOKKOS_INLINE_FUNCTION explicit D(int i) : B(i) {}
};

// TODO: add a Kokkos::pair device test when adding kokkos pair constructors
TEST(host_tuple_cnstr, move_pair) {
  {
    typedef std::pair<long, std::unique_ptr<D>> T0;
    typedef cexa::tuple<long long, std::unique_ptr<B>> T1;
    T0 t0(2, std::unique_ptr<D>(new D(3)));
    T1 t1 = std::move(t0);
    CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
    CEXA_EXPECT_EQ(cexa::get<1>(t1)->id_, 3);
  }

  {
    using pair_t = std::pair<int, char>;
    constexpr cexa::tuple<long, long> t(pair_t(0, 'a'));
    static_assert(cexa::get<0>(t) == 0, "");
    static_assert(cexa::get<1>(t) == 'a', "");
  }
}
