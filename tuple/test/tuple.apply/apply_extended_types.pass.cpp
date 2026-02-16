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

// UNSUPPORTED: c++03, c++11, c++14

// <tuple>

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&)

// Testing extended function types. The extended function types are those
// named by INVOKE but that are not actual callable objects. These include
// bullets 1-4 of invoke.

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

#if defined(CEXA_ON_DEVICE)
__device__ int count = 0;
#else
int count = 0;
#endif

struct A_int_0
{
    KOKKOS_INLINE_FUNCTION A_int_0() : obj1(0){}
    KOKKOS_INLINE_FUNCTION A_int_0(int x) : obj1(x) {}
    KOKKOS_INLINE_FUNCTION int mem1() { return ++count; }
    KOKKOS_INLINE_FUNCTION int mem2() const { return ++count; }
    int const obj1;
};

struct A_int_1
{
    KOKKOS_INLINE_FUNCTION A_int_1() {}
    KOKKOS_INLINE_FUNCTION A_int_1(int) {}
    KOKKOS_INLINE_FUNCTION int mem1(int x) { return count += x; }
    KOKKOS_INLINE_FUNCTION int mem2(int x) const { return count += x; }
};

struct A_int_2
{
    KOKKOS_INLINE_FUNCTION A_int_2() {}
    KOKKOS_INLINE_FUNCTION A_int_2(int) {}
    KOKKOS_INLINE_FUNCTION int mem1(int x, int y) { return count += (x + y); }
    KOKKOS_INLINE_FUNCTION int mem2(int x, int y) const { return count += (x + y); }
};

template <class A>
struct A_wrap
{
    KOKKOS_INLINE_FUNCTION A_wrap() {}
    KOKKOS_INLINE_FUNCTION A_wrap(int x) : m_a(x) {}
    KOKKOS_INLINE_FUNCTION A & operator*() { return m_a; }
    KOKKOS_INLINE_FUNCTION A const & operator*() const { return m_a; }
    A m_a;
};

typedef A_wrap<A_int_0> A_wrap_0;
typedef A_wrap<A_int_1> A_wrap_1;
typedef A_wrap<A_int_2> A_wrap_2;


template <class A>
struct A_base : public A
{
    KOKKOS_INLINE_FUNCTION A_base() : A() {}
    KOKKOS_INLINE_FUNCTION A_base(int x) : A(x) {}
};

typedef A_base<A_int_0> A_base_0;
typedef A_base<A_int_1> A_base_1;
typedef A_base<A_int_2> A_base_2;


template <
    class Tuple, class ConstTuple
  , class TuplePtr, class ConstTuplePtr
  , class TupleWrap, class ConstTupleWrap
  , class TupleBase, class ConstTupleBase
  >
KOKKOS_INLINE_FUNCTION void test_ext_int_0()
{
    count = 0;
    typedef A_int_0 T;
    typedef A_wrap_0 Wrap;
    typedef A_base_0 Base;

    typedef int(T::*mem1_t)();
    mem1_t mem1 = &T::mem1;

    typedef int(T::*mem2_t)() const;
    mem2_t mem2 = &T::mem2;

    typedef int const T::*obj1_t;
    obj1_t obj1 = &T::obj1;

    // member function w/ref
    {
        T a;
        Tuple t{a};
        CEXA_EXPECT_EQ(1, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 1);
    }
    count = 0;
    // member function w/pointer
    {
        T a;
        TuplePtr t{&a};
        CEXA_EXPECT_EQ(1, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 1);
    }
    count = 0;
    // member function w/base
    {
        Base a;
        TupleBase t{a};
        CEXA_EXPECT_EQ(1, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 1);
    }
    count = 0;
    // member function w/wrap
    {
        Wrap a;
        TupleWrap t{a};
        CEXA_EXPECT_EQ(1, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 1);
    }
    count = 0;
    // const member function w/ref
    {
        T const a;
        ConstTuple t{a};
        CEXA_EXPECT_EQ(1, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 1);
    }
    count = 0;
    // const member function w/pointer
    {
        T const a;
        ConstTuplePtr t{&a};
        CEXA_EXPECT_EQ(1, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 1);
    }
    count = 0;
    // const member function w/base
    {
        Base const a;
        ConstTupleBase t{a};
        CEXA_EXPECT_EQ(1, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 1);
    }
    count = 0;
    // const member function w/wrapper
    {
        Wrap const a;
        ConstTupleWrap t{a};
        CEXA_EXPECT_EQ(1, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(1, count);
    }
    // member object w/ref
    {
        T a{42};
        Tuple t{a};
        CEXA_EXPECT_EQ(42, cexa::apply(obj1, t));
    }
    // member object w/pointer
    {
        T a{42};
        TuplePtr t{&a};
        CEXA_EXPECT_EQ(42, cexa::apply(obj1, t));
    }
    // member object w/base
    {
        Base a{42};
        TupleBase t{a};
        CEXA_EXPECT_EQ(42, cexa::apply(obj1, t));
    }
    // member object w/wrapper
    {
        Wrap a{42};
        TupleWrap t{a};
        CEXA_EXPECT_EQ(42, cexa::apply(obj1, t));
    }
}


template <
    class Tuple, class ConstTuple
  , class TuplePtr, class ConstTuplePtr
  , class TupleWrap, class ConstTupleWrap
  , class TupleBase, class ConstTupleBase
  >
KOKKOS_INLINE_FUNCTION void test_ext_int_1()
{
    count = 0;
    typedef A_int_1 T;
    typedef A_wrap_1 Wrap;
    typedef A_base_1 Base;

    typedef int(T::*mem1_t)(int);
    mem1_t mem1 = &T::mem1;

    typedef int(T::*mem2_t)(int) const;
    mem2_t mem2 = &T::mem2;

    // member function w/ref
    {
        T a;
        Tuple t{a, 2};
        CEXA_EXPECT_EQ(2, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 2);
    }
    count = 0;
    // member function w/pointer
    {
        T a;
        TuplePtr t{&a, 3};
        CEXA_EXPECT_EQ(3, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 3);
    }
    count = 0;
    // member function w/base
    {
        Base a;
        TupleBase t{a, 4};
        CEXA_EXPECT_EQ(4, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 4);
    }
    count = 0;
    // member function w/wrap
    {
        Wrap a;
        TupleWrap t{a, 5};
        CEXA_EXPECT_EQ(5, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 5);
    }
    count = 0;
    // const member function w/ref
    {
        T const a;
        ConstTuple t{a, 6};
        CEXA_EXPECT_EQ(6, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 6);
    }
    count = 0;
    // const member function w/pointer
    {
        T const a;
        ConstTuplePtr t{&a, 7};
        CEXA_EXPECT_EQ(7, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 7);
    }
    count = 0;
    // const member function w/base
    {
        Base const a;
        ConstTupleBase t{a, 8};
        CEXA_EXPECT_EQ(8, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 8);
    }
    count = 0;
    // const member function w/wrapper
    {
        Wrap const a;
        ConstTupleWrap t{a, 9};
        CEXA_EXPECT_EQ(9, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(9, count);
    }
}


template <
    class Tuple, class ConstTuple
  , class TuplePtr, class ConstTuplePtr
  , class TupleWrap, class ConstTupleWrap
  , class TupleBase, class ConstTupleBase
  >
KOKKOS_INLINE_FUNCTION void test_ext_int_2()
{
    count = 0;
    typedef A_int_2 T;
    typedef A_wrap_2 Wrap;
    typedef A_base_2 Base;

    typedef int(T::*mem1_t)(int, int);
    mem1_t mem1 = &T::mem1;

    typedef int(T::*mem2_t)(int, int) const;
    mem2_t mem2 = &T::mem2;

    // member function w/ref
    {
        T a;
        Tuple t{a, 1, 1};
        CEXA_EXPECT_EQ(2, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 2);
    }
    count = 0;
    // member function w/pointer
    {
        T a;
        TuplePtr t{&a, 1, 2};
        CEXA_EXPECT_EQ(3, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 3);
    }
    count = 0;
    // member function w/base
    {
        Base a;
        TupleBase t{a, 2, 2};
        CEXA_EXPECT_EQ(4, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 4);
    }
    count = 0;
    // member function w/wrap
    {
        Wrap a;
        TupleWrap t{a, 2, 3};
        CEXA_EXPECT_EQ(5, cexa::apply(mem1, t));
        CEXA_EXPECT_EQ(count, 5);
    }
    count = 0;
    // const member function w/ref
    {
        T const a;
        ConstTuple t{a, 3, 3};
        CEXA_EXPECT_EQ(6, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 6);
    }
    count = 0;
    // const member function w/pointer
    {
        T const a;
        ConstTuplePtr t{&a, 3, 4};
        CEXA_EXPECT_EQ(7, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 7);
    }
    count = 0;
    // const member function w/base
    {
        Base const a;
        ConstTupleBase t{a, 4, 4};
        CEXA_EXPECT_EQ(8, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(count, 8);
    }
    count = 0;
    // const member function w/wrapper
    {
        Wrap const a;
        ConstTupleWrap t{a, 4, 5};
        CEXA_EXPECT_EQ(9, cexa::apply(mem2, t));
        CEXA_EXPECT_EQ(9, count);
    }
}

// clang-foramt off
CEXA_TEST(tuple_apply, apply_extended_types, (
    {
        test_ext_int_0<
            cexa::tuple<A_int_0 &>, cexa::tuple<A_int_0 const &>
          , cexa::tuple<A_int_0 *>, cexa::tuple<A_int_0 const *>
          , cexa::tuple<A_wrap_0 &>, cexa::tuple<A_wrap_0 const &>
          , cexa::tuple<A_base_0 &>, cexa::tuple<A_base_0 const &>
          >();
        test_ext_int_0<
            cexa::tuple<A_int_0>, cexa::tuple<A_int_0 const>
          , cexa::tuple<A_int_0 *>, cexa::tuple<A_int_0 const *>
          , cexa::tuple<A_wrap_0>, cexa::tuple<A_wrap_0 const>
          , cexa::tuple<A_base_0>, cexa::tuple<A_base_0 const>
          >();
        // test_ext_int_0<
        //     std::array<A_int_0, 1>, std::array<A_int_0 const, 1>
        //   , std::array<A_int_0*, 1>, std::array<A_int_0 const*, 1>
        //   , std::array<A_wrap_0, 1>, std::array<A_wrap_0 const, 1>
        //   , std::array<A_base_0, 1>, std::array<A_base_0 const, 1>
        //   >();
    }
    {
        test_ext_int_1<
            cexa::tuple<A_int_1 &, int>, cexa::tuple<A_int_1 const &, int>
          , cexa::tuple<A_int_1 *, int>, cexa::tuple<A_int_1 const *, int>
          , cexa::tuple<A_wrap_1 &, int>, cexa::tuple<A_wrap_1 const &, int>
          , cexa::tuple<A_base_1 &, int>, cexa::tuple<A_base_1 const &, int>
          >();
        test_ext_int_1<
            cexa::tuple<A_int_1, int>, cexa::tuple<A_int_1 const, int>
          , cexa::tuple<A_int_1 *, int>, cexa::tuple<A_int_1 const *, int>
          , cexa::tuple<A_wrap_1, int>, cexa::tuple<A_wrap_1 const, int>
          , cexa::tuple<A_base_1, int>, cexa::tuple<A_base_1 const, int>
          >();
        // test_ext_int_1<
        //     std::pair<A_int_1 &, int>, std::pair<A_int_1 const &, int>
        //   , std::pair<A_int_1 *, int>, std::pair<A_int_1 const *, int>
        //   , std::pair<A_wrap_1 &, int>, std::pair<A_wrap_1 const &, int>
        //   , std::pair<A_base_1 &, int>, std::pair<A_base_1 const &, int>
        //   >();
        // test_ext_int_1<
        //     std::pair<A_int_1, int>, std::pair<A_int_1 const, int>
        //   , std::pair<A_int_1 *, int>, std::pair<A_int_1 const *, int>
        //   , std::pair<A_wrap_1, int>, std::pair<A_wrap_1 const, int>
        //   , std::pair<A_base_1, int>, std::pair<A_base_1 const, int>
        //   >();
    }
    {
        test_ext_int_2<
            cexa::tuple<A_int_2 &, int, int>, cexa::tuple<A_int_2 const &, int, int>
          , cexa::tuple<A_int_2 *, int, int>, cexa::tuple<A_int_2 const *, int, int>
          , cexa::tuple<A_wrap_2 &, int, int>, cexa::tuple<A_wrap_2 const &, int, int>
          , cexa::tuple<A_base_2 &, int, int>, cexa::tuple<A_base_2 const &, int, int>
          >();
        test_ext_int_2<
            cexa::tuple<A_int_2, int, int>, cexa::tuple<A_int_2 const, int, int>
          , cexa::tuple<A_int_2 *, int, int>, cexa::tuple<A_int_2 const *, int, int>
          , cexa::tuple<A_wrap_2, int, int>, cexa::tuple<A_wrap_2 const, int, int>
          , cexa::tuple<A_base_2, int, int>, cexa::tuple<A_base_2 const, int, int>
          >();
    }
))
// clang-foramt on
