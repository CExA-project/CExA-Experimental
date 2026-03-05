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

// template <class... UTypes> tuple(tuple<UTypes...>&& u);

// UNSUPPORTED: c++03

#include <memory>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

struct Explicit {
  int value;
  KOKKOS_INLINE_FUNCTION explicit Explicit(int x) : value(x) {}
};

struct Implicit {
  int value;
  KOKKOS_INLINE_FUNCTION Implicit(int x) : value(x) {}
};

struct B
{
    int id_;

    KOKKOS_INLINE_FUNCTION explicit B(int i) : id_(i) {}
    KOKKOS_DEFAULTED_FUNCTION B(const B&) = default;
    KOKKOS_DEFAULTED_FUNCTION B& operator=(const B&) = default;
    KOKKOS_INLINE_FUNCTION virtual ~B() {}
};

struct D
    : B
{
    KOKKOS_INLINE_FUNCTION explicit D(int i) : B(i) {}
};

struct BonkersBananas {
  template <class T>
  operator T() &&;
  template <class T, class = void>
  explicit operator T() && = delete;
};

namespace test1 {
  using ReturnType = cexa::tuple<int, int>;
  static_assert(std::is_convertible<BonkersBananas, ReturnType>(), "");
// FIXME: investigate why this static_assert fails on nvcc and hip
#if !(defined(KOKKOS_COMPILER_NVCC) || defined(KOKKOS_ENABLE_HIP))
  static_assert(!std::is_constructible<ReturnType, BonkersBananas>(), "");
#endif
}

// clang-format off
CEXA_TEST(tuple_cnstr, convert_move, (
    {
        typedef cexa::tuple<long> T0;
        typedef cexa::tuple<long long> T1;
        T0 t0(2);
        T1 t1 = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
    }
    {
        typedef cexa::tuple<long, char> T0;
        typedef cexa::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1 = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
    }
    {
        typedef cexa::tuple<long, char, D> T0;
        typedef cexa::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1 = std::move(t0);
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
        CEXA_EXPECT_EQ(cexa::get<2>(t1).id_, 3);
    }
    {
        D d(3);
        typedef cexa::tuple<long, char, D&> T0;
        typedef cexa::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d);
        T1 t1 = std::move(t0);
        d.id_ = 2;
        CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
        CEXA_EXPECT_EQ(cexa::get<2>(t1).id_, 2);
    }
    {
        cexa::tuple<int> t1(42);
        cexa::tuple<Explicit> t2(std::move(t1));
        CEXA_EXPECT_EQ(cexa::get<0>(t2).value, 42);
    }
    {
        cexa::tuple<int> t1(42);
        cexa::tuple<Implicit> t2 = std::move(t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t2).value, 42);
    }
))

    CEXA_HOST_DEVICE_NVCC_WARNINGS_PUSH()
TEST(tuple_cnstr, convert_move_host) {
    typedef cexa::tuple<long, char, std::unique_ptr<D>> T0;
    typedef cexa::tuple<long long, int, std::unique_ptr<B>> T1;
    T0 t0(2, 'a', std::unique_ptr<D>(new D(3)));
    T1 t1 = std::move(t0);
    CEXA_EXPECT_EQ(cexa::get<0>(t1), 2);
    CEXA_EXPECT_EQ(cexa::get<1>(t1), int('a'));
    CEXA_EXPECT_EQ(cexa::get<2>(t1)->id_, 3);
}
    CEXA_HOST_DEVICE_NVCC_WARNINGS_POP()
// clang-format on
