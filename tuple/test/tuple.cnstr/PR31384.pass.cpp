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

// UNSUPPORTED: c++03

// FIXME: Why does this start to fail with GCC 14?
// XFAIL: !(c++11 || c++14) && gcc-15

// See https://llvm.org/PR31384.

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

#if !defined(KOKKOS_COMPILER_GNU) || (KOKKOS_COMPILER_GNU < 14)
#if defined(CEXA_ON_DEVICE)
__device__ int count = 0;
#else
int count = 0;
#endif

struct Explicit {
  KOKKOS_DEFAULTED_FUNCTION Explicit() = default;
  KOKKOS_INLINE_FUNCTION explicit Explicit(int) {}
};

struct Implicit {
  KOKKOS_DEFAULTED_FUNCTION Implicit() = default;
  KOKKOS_INLINE_FUNCTION Implicit(int) {}
};

template<class T>
struct Derived : cexa::tuple<T> {
  using cexa::tuple<T>::tuple;
  template<class U>
  KOKKOS_INLINE_FUNCTION operator cexa::tuple<U>() && { ++count; return {}; }
};


template<class T>
struct ExplicitDerived : cexa::tuple<T> {
  using cexa::tuple<T>::tuple;
  template<class U>
  KOKKOS_INLINE_FUNCTION explicit operator cexa::tuple<U>() && { ++count; return {}; }
};

// clang-format off
CEXA_TEST(tuple_cnstr, PR31384, (
  {
    cexa::tuple<Explicit> foo = Derived<int>{42}; ((void)foo);
    CEXA_EXPECT_EQ(count, 1);
    // FIXME: This fails with nvcc, the element-wise conversion constructor is chosen
    #if !defined(KOKKOS_COMPILER_NVCC)
    Derived<int> d{42};
    cexa::tuple<Explicit> bar(std::move(d)); ((void)bar);
    CEXA_EXPECT_EQ(count, 2);
    #endif
  }
  count = 0;
  {
    cexa::tuple<Implicit> foo = Derived<int>{42}; ((void)foo);
    CEXA_EXPECT_EQ(count, 1);
    // FIXME: This fails with nvcc, the element-wise conversion constructor is chosen
    #if !defined(KOKKOS_COMPILER_NVCC)
    Derived<int> d{42};
    cexa::tuple<Implicit> bar(std::move(d)); ((void)bar);
    CEXA_EXPECT_EQ(count, 2);
    #endif
  }
  count = 0;
  {
    static_assert(!std::is_convertible<ExplicitDerived<int>, cexa::tuple<Explicit>>::value, "");
    // FIXME: This fails with nvcc, the element-wise conversion constructor is chosen
    #if !defined(KOKKOS_COMPILER_NVCC)
    ExplicitDerived<int> d{42};
    cexa::tuple<Explicit> bar(std::move(d)); ((void)bar);
    CEXA_EXPECT_EQ(count, 1);
    #endif
  }
  count = 0;
  {
    cexa::tuple<Implicit> foo = ExplicitDerived<int>{42}; ((void)foo);
    static_assert(std::is_convertible<ExplicitDerived<int>, cexa::tuple<Implicit>>::value, "");
    CEXA_EXPECT_EQ(count, 0);
    // FIXME: This fails with nvcc, the element-wise conversion constructor is chosen
    #if !defined(KOKKOS_COMPILER_NVCC)
    ExplicitDerived<int> d{42};
    cexa::tuple<Implicit> bar(std::move(d)); ((void)bar);
    CEXA_EXPECT_EQ(count, 1);
    #endif
  }
  count = 0;
))
// clang-format on
#endif
