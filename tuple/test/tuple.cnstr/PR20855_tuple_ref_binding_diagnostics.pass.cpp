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

// <tuple>

// See https://llvm.org/PR20855.

#include <functional>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>

template <class Tp>
struct ConvertsTo {
  using RawTp = typename std::remove_cv< typename std::remove_reference<Tp>::type>::type;

  KOKKOS_INLINE_FUNCTION operator Tp() const {
    return static_cast<Tp>(value);
  }

  mutable RawTp value;
};

struct Base {};
struct Derived : Base {};

static_assert(std::is_constructible<int&, std::reference_wrapper<int>>::value, "");
static_assert(std::is_constructible<int const&, std::reference_wrapper<int>>::value, "");

template <class T> struct CannotDeduce {
 using type = T;
};

template <class ...Args>
KOKKOS_INLINE_FUNCTION void F(typename CannotDeduce<cexa::tuple<Args...>>::type const&) {}

KOKKOS_INLINE_FUNCTION void compile_tests() {
  {
    F<int, int const&>(cexa::make_tuple(42, 42));
  }
  {
    F<int, int const&>(cexa::make_tuple<const int&, const int&>(42, 42));
    cexa::tuple<int, int const&> t(cexa::make_tuple<const int&, const int&>(42, 42));
  }
  {
    // TODO: replace string by a device-compatible type
    int* ptr = nullptr;
    auto fn = &F<int, Kokkos::View<int*> const&>;
    fn(cexa::tuple<int, Kokkos::View<int*> const&>(42, Kokkos::View<int*>(ptr, 3)));
    fn(cexa::make_tuple(42, Kokkos::View<int*>(ptr, 3)));
  }
  {
    Derived d;
    cexa::tuple<Base&, Base const&> t(d, d);
  }
  {
    ConvertsTo<int&> ct;
    cexa::tuple<int, int&> t(42, ct);
  }
}

// TODO: add when adding allocator constructors
// void allocator_tests() {
//     std::allocator<int> alloc;
//     int x = 42;
//     {
//         cexa::tuple<int&> t(std::ref(x));
//         CEXA_EXPECT_EQ(&cexa::get<0>(t), &x);
//         cexa::tuple<int&> t1(std::allocator_arg, alloc, std::ref(x));
//         CEXA_EXPECT_EQ(&cexa::get<0>(t1), &x);
//     }
//     {
//         auto r = std::ref(x);
//         auto const& cr = r;
//         cexa::tuple<int&> t(r);
//         CEXA_EXPECT_EQ(&cexa::get<0>(t), &x);
//         cexa::tuple<int&> t1(cr);
//         CEXA_EXPECT_EQ(&cexa::get<0>(t1), &x);
//         cexa::tuple<int&> t2(std::allocator_arg, alloc, r);
//         CEXA_EXPECT_EQ(&cexa::get<0>(t2), &x);
//         cexa::tuple<int&> t3(std::allocator_arg, alloc, cr);
//         CEXA_EXPECT_EQ(&cexa::get<0>(t3), &x);
//     }
//     {
//         cexa::tuple<int const&> t(std::ref(x));
//         CEXA_EXPECT_EQ(&cexa::get<0>(t), &x);
//         cexa::tuple<int const&> t2(std::cref(x));
//         CEXA_EXPECT_EQ(&cexa::get<0>(t2), &x);
//         cexa::tuple<int const&> t3(std::allocator_arg, alloc, std::ref(x));
//         CEXA_EXPECT_EQ(&cexa::get<0>(t3), &x);
//         cexa::tuple<int const&> t4(std::allocator_arg, alloc, std::cref(x));
//         CEXA_EXPECT_EQ(&cexa::get<0>(t4), &x);
//     }
//     {
//         auto r = std::ref(x);
//         auto cr = std::cref(x);
//         cexa::tuple<int const&> t(r);
//         CEXA_EXPECT_EQ(&cexa::get<0>(t), &x);
//         cexa::tuple<int const&> t2(cr);
//         CEXA_EXPECT_EQ(&cexa::get<0>(t2), &x);
//         cexa::tuple<int const&> t3(std::allocator_arg, alloc, r);
//         CEXA_EXPECT_EQ(&cexa::get<0>(t3), &x);
//         cexa::tuple<int const&> t4(std::allocator_arg, alloc, cr);
//         CEXA_EXPECT_EQ(&cexa::get<0>(t4), &x);
//     }
// }


int main(int, char**) {
  compile_tests();
  // allocator_tests();

  return 0;
}
