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
//
// UNSUPPORTED: c++03

// <tuple>

// template <class... Types> class tuple;

// template <class ...UTypes>
//    EXPLICIT(...) tuple(UTypes&&...)

// Check that the UTypes... ctor is properly disabled before evaluating any
// SFINAE when the copy/move ctor from another tuple should clearly be selected
// instead. This happens 'sizeof...(UTypes) == 1' and the first element of
// 'UTypes...' is an instance of the tuple itself.
//
// See https://llvm.org/PR23256.

#include <memory>
#include <type_traits>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>


struct UnconstrainedCtor {
  int value_;

  KOKKOS_INLINE_FUNCTION UnconstrainedCtor() : value_(0) {}

  // Blows up when instantiated for any type other than int. Because the ctor
  // is constexpr it is instantiated by 'is_constructible' and 'is_convertible'
  // for Clang based compilers. GCC does not instantiate the ctor body
  // but it does instantiate the noexcept specifier and it will blow up there.
  template <typename T>
  KOKKOS_INLINE_FUNCTION constexpr UnconstrainedCtor(T value) noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
      static_assert(std::is_same<int, T>::value, "");
  }
};

struct ExplicitUnconstrainedCtor {
  int value_;

  KOKKOS_INLINE_FUNCTION ExplicitUnconstrainedCtor() : value_(0) {}

  template <typename T>
  KOKKOS_INLINE_FUNCTION constexpr explicit ExplicitUnconstrainedCtor(T value)
    noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
      static_assert(std::is_same<int, T>::value, "");
  }

};

// clang-format off
CEXA_TEST(tuple_cnstr, PR23256_constrain_UTypes_ctor, (
    typedef UnconstrainedCtor A;
    typedef ExplicitUnconstrainedCtor ExplicitA;
    {
        static_assert(std::is_copy_constructible<cexa::tuple<A>>::value, "");
        static_assert(std::is_move_constructible<cexa::tuple<A>>::value, "");
        static_assert(std::is_copy_constructible<cexa::tuple<ExplicitA>>::value, "");
        static_assert(std::is_move_constructible<cexa::tuple<ExplicitA>>::value, "");
    }
    // TODO: requires allocator constructors
    // {
    //     static_assert(std::is_constructible<
    //         cexa::tuple<A>,
    //         std::allocator_arg_t, std::allocator<int>,
    //         cexa::tuple<A> const&
    //     >::value, "");
    //     static_assert(std::is_constructible<
    //         cexa::tuple<A>,
    //         std::allocator_arg_t, std::allocator<int>,
    //         cexa::tuple<A> &&
    //     >::value, "");
    //     static_assert(std::is_constructible<
    //         cexa::tuple<ExplicitA>,
    //         std::allocator_arg_t, std::allocator<int>,
    //         cexa::tuple<ExplicitA> const&
    //     >::value, "");
    //     static_assert(std::is_constructible<
    //         cexa::tuple<ExplicitA>,
    //         std::allocator_arg_t, std::allocator<int>,
    //         cexa::tuple<ExplicitA> &&
    //     >::value, "");
    // }
    {
        cexa::tuple<A&&> t(cexa::forward_as_tuple(A{}));
        ((void)t);
        cexa::tuple<ExplicitA&&> t2(cexa::forward_as_tuple(ExplicitA{}));
        ((void)t2);
    }
))
// clang-format on
