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

// This test ensures that cexa::tuple is lazy when it comes to checking whether
// the elements it is assigned from can be used to assign to the types in
// the tuple.

#include <tuple.hpp>
#include <Kokkos_Macros.hpp>
#include <array>

template <bool Enable, class ...Class>
KOKKOS_INLINE_FUNCTION constexpr typename std::enable_if<Enable, bool>::type BlowUp() {
  static_assert(Enable && sizeof...(Class) != sizeof...(Class), "");
  return true;
}

template<class T>
struct Fail {
  static_assert(sizeof(T) != sizeof(T), "");
  using type = void;
};

struct NoAssign {
  KOKKOS_DEFAULTED_FUNCTION NoAssign() = default;
  KOKKOS_DEFAULTED_FUNCTION NoAssign(NoAssign const&) = default;
  template <class T, class = typename std::enable_if<sizeof(T) != sizeof(T)>::type>
  KOKKOS_INLINE_FUNCTION NoAssign& operator=(T) { return *this; }
};

template <int>
struct DieOnAssign {
  KOKKOS_DEFAULTED_FUNCTION DieOnAssign() = default;
  template <class T, class X = typename std::enable_if<!std::is_same<T, DieOnAssign>::value>::type,
                     class = typename Fail<X>::type>
  KOKKOS_INLINE_FUNCTION DieOnAssign& operator=(T) {
    return *this;
  }
};

KOKKOS_INLINE_FUNCTION void test_arity_checks() {
  {
    using T = cexa::tuple<int, DieOnAssign<0>, int>;
    using P = std::pair<int, int>;
    static_assert(!std::is_assignable<T&, P const&>::value, "");
  }
  {
    using T = cexa::tuple<int, int, DieOnAssign<1> >;
    using A = std::array<int, 1>;
    static_assert(!std::is_assignable<T&, A const&>::value, "");
  }
}

KOKKOS_INLINE_FUNCTION void test_assignability_checks() {
  {
    using T1 = cexa::tuple<int, NoAssign, DieOnAssign<2> >;
    using T2 = cexa::tuple<long, long, long>;
    static_assert(!std::is_assignable<T1&, T2 const&>::value, "");
  }
  {
    using T1 = cexa::tuple<NoAssign, DieOnAssign<3> >;
    using T2 = std::pair<long, double>;
    static_assert(!std::is_assignable<T1&, T2 const&>::value, "");
  }
}

int main(int, char**) {
  test_arity_checks();
  test_assignability_checks();
  return 0;
}
