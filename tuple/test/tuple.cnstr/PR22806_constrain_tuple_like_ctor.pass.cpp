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

// Check that the tuple-like ctors are properly disabled when the UTypes...
// constructor should be selected.
//
// See https://llvm.org/PR22806.

#include <memory>
#include <type_traits>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

template <class Tp>
using uncvref_t = typename std::remove_cv<typename std::remove_reference<Tp>::type>::type;

template <class Tuple, class = uncvref_t<Tuple>>
struct IsTuple : std::false_type {};

template <class Tuple, class ...Args>
struct IsTuple<Tuple, cexa::tuple<Args...>> : std::true_type {};

struct ConstructibleFromTupleAndInt {
  enum State { FromTuple, FromInt, Copied, Moved };
  State state;

  KOKKOS_INLINE_FUNCTION ConstructibleFromTupleAndInt(ConstructibleFromTupleAndInt const&) : state(Copied) {}
  KOKKOS_INLINE_FUNCTION ConstructibleFromTupleAndInt(ConstructibleFromTupleAndInt &&) : state(Moved) {}

  template <class Tuple, class = typename std::enable_if<IsTuple<Tuple>::value>::type>
  KOKKOS_INLINE_FUNCTION explicit ConstructibleFromTupleAndInt(Tuple&&) : state(FromTuple) {}

  KOKKOS_INLINE_FUNCTION explicit ConstructibleFromTupleAndInt(int) : state(FromInt) {}
};

struct ConvertibleFromTupleAndInt {
  enum State { FromTuple, FromInt, Copied, Moved };
  State state;

  KOKKOS_INLINE_FUNCTION ConvertibleFromTupleAndInt(ConvertibleFromTupleAndInt const&) : state(Copied) {}
  KOKKOS_INLINE_FUNCTION ConvertibleFromTupleAndInt(ConvertibleFromTupleAndInt &&) : state(Moved) {}

  template <class Tuple, class = typename std::enable_if<IsTuple<Tuple>::value>::type>
  KOKKOS_INLINE_FUNCTION ConvertibleFromTupleAndInt(Tuple&&) : state(FromTuple) {}

  KOKKOS_INLINE_FUNCTION ConvertibleFromTupleAndInt(int) : state(FromInt) {}
};

struct ConstructibleFromInt {
  enum State { FromInt, Copied, Moved };
  State state;

  KOKKOS_INLINE_FUNCTION ConstructibleFromInt(ConstructibleFromInt const&) : state(Copied) {}
  KOKKOS_INLINE_FUNCTION ConstructibleFromInt(ConstructibleFromInt &&) : state(Moved) {}

  KOKKOS_INLINE_FUNCTION explicit ConstructibleFromInt(int) : state(FromInt) {}
};

struct ConvertibleFromInt {
  enum State { FromInt, Copied, Moved };
  State state;

  KOKKOS_INLINE_FUNCTION ConvertibleFromInt(ConvertibleFromInt const&) : state(Copied) {}
  KOKKOS_INLINE_FUNCTION ConvertibleFromInt(ConvertibleFromInt &&) : state(Moved) {}
  KOKKOS_INLINE_FUNCTION ConvertibleFromInt(int) : state(FromInt) {}
};

// clang-format off
CEXA_TEST(tuple_cnstr, PR22806_constrain_tuple_like_ctor, (
    // Test for the creation of dangling references when a tuple is used to
    // store a reference to another tuple as its only element.
    // Ex cexa::tuple<cexa::tuple<int>&&>.
    // In this case the constructors 1) 'tuple(UTypes&&...)'
    // and 2) 'tuple(TupleLike&&)' need to be manually disambiguated because
    // when both #1 and #2 participate in partial ordering #2 will always
    // be chosen over #1.
    // See PR22806  and LWG issue #2549 for more information.
    // (https://llvm.org/PR22806)
    using T = cexa::tuple<int>;
    // std::allocator<int> A;
    { // rvalue reference
        T t1(42);
        cexa::tuple< T&& > t2(std::move(t1));
        CEXA_EXPECT_EQ(&cexa::get<0>(t2), &t1);
    }
    { // const lvalue reference
        T t1(42);

        cexa::tuple< T const & > t2(t1);
        CEXA_EXPECT_EQ(&cexa::get<0>(t2), &t1);

        cexa::tuple< T const & > t3(static_cast<T const&>(t1));
        CEXA_EXPECT_EQ(&cexa::get<0>(t3), &t1);
    }
    { // lvalue reference
        T t1(42);

        cexa::tuple< T & > t2(t1);
        CEXA_EXPECT_EQ(&cexa::get<0>(t2), &t1);
    }
    { // const rvalue reference
        T t1(42);

        cexa::tuple< T const && > t2(std::move(t1));
        CEXA_EXPECT_EQ(&cexa::get<0>(t2), &t1);
    }
    // TODO: enable when adding allocator constructors
    // { // rvalue reference via uses-allocator
    //     T t1(42);
    //     cexa::tuple< T&& > t2(std::allocator_arg, A, std::move(t1));
    //     CEXA_EXPECT_EQ(&cexa::get<0>(t2), &t1);
    // }
    // { // const lvalue reference via uses-allocator
    //     T t1(42);
    //
    //     cexa::tuple< T const & > t2(std::allocator_arg, A, t1);
    //     CEXA_EXPECT_EQ(&cexa::get<0>(t2), &t1);
    //
    //     cexa::tuple< T const & > t3(std::allocator_arg, A, static_cast<T const&>(t1));
    //     CEXA_EXPECT_EQ(&cexa::get<0>(t3), &t1);
    // }
    // { // lvalue reference via uses-allocator
    //     T t1(42);
    //
    //     cexa::tuple< T & > t2(std::allocator_arg, A, t1);
    //     CEXA_EXPECT_EQ(&cexa::get<0>(t2), &t1);
    // }
    // { // const rvalue reference via uses-allocator
    //     T const t1(42);
    //     cexa::tuple< T const && > t2(std::allocator_arg, A, std::move(t1));
    //     CEXA_EXPECT_EQ(&cexa::get<0>(t2), &t1);
    // }
    // Test constructing a 1-tuple of the form tuple<UDT> from another 1-tuple
    // 'tuple<T>' where UDT *can* be constructed from 'tuple<T>'. In this case
    // the 'tuple(UTypes...)' ctor should be chosen and 'UDT' constructed from
    // 'tuple<T>'.
    {
        using VT = ConstructibleFromTupleAndInt;
        cexa::tuple<int> t1(42);
        cexa::tuple<VT> t2(t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t2).state, VT::FromTuple);
    }
    {
        using VT = ConvertibleFromTupleAndInt;
        cexa::tuple<int> t1(42);
        cexa::tuple<VT> t2 = {t1};
        CEXA_EXPECT_EQ(cexa::get<0>(t2).state, VT::FromTuple);
    }
    // Test constructing a 1-tuple of the form tuple<UDT> from another 1-tuple
    // 'tuple<T>' where UDT cannot be constructed from 'tuple<T>' but can
    // be constructed from 'T'. In this case the tuple-like ctor should be
    // chosen and 'UDT' constructed from 'T'
    {
        using VT = ConstructibleFromInt;
        cexa::tuple<int> t1(42);
        cexa::tuple<VT> t2(t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t2).state, VT::FromInt);
    }
    {
        using VT = ConvertibleFromInt;
        cexa::tuple<int> t1(42);
        cexa::tuple<VT> t2 = {t1};
        CEXA_EXPECT_EQ(cexa::get<0>(t2).state, VT::FromInt);
    }
))
// clang-format on
