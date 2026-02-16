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

// Stress testing large arities with tuple and array.

#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

////////////////////////////////////////////////////////////////////////////////
template <class T, std::size_t Dummy = 0>
struct always_imp
{
    typedef T type;
};

template <class T, std::size_t Dummy = 0>
using always_t = typename always_imp<T, Dummy>::type;

////////////////////////////////////////////////////////////////////////////////
template <class Tuple, class Idx>
struct make_function;

template <class Tp, std::size_t ...Idx>
struct make_function<Tp, std::integer_sequence<std::size_t, Idx...>>
{
    using type = bool (*)(always_t<Tp, Idx>...);
};

template <class Tp, std::size_t Size>
using make_function_t = typename make_function<Tp, std::make_index_sequence<Size>>::type;

////////////////////////////////////////////////////////////////////////////////
template <class Tp, class Idx>
struct make_tuple_imp;

////////////////////////////////////////////////////////////////////////////////
template <class Tp, std::size_t ...Idx>
struct make_tuple_imp<Tp, std::integer_sequence<std::size_t, Idx...>>
{
    using type = cexa::tuple<always_t<Tp, Idx>...>;
};

template <class Tp, std::size_t Size>
using make_tuple_t = typename make_tuple_imp<Tp, std::make_index_sequence<Size>>::type;

template <class ...Types>
KOKKOS_INLINE_FUNCTION bool test_apply_fn(Types...) { return true; }


template <std::size_t Size>
KOKKOS_INLINE_FUNCTION void test_all()
{

    // using A = std::array<int, Size>;
    // using ConstA = std::array<int const, Size>;

    using Tuple = make_tuple_t<int, Size>;
    using CTuple = make_tuple_t<const int, Size>;

    using ValFn  = make_function_t<int, Size>;
    ValFn val_fn = &test_apply_fn;

    using RefFn  = make_function_t<int &, Size>;
    RefFn ref_fn = &test_apply_fn;

    using CRefFn = make_function_t<int const &, Size>;
    CRefFn cref_fn = &test_apply_fn;

    using RRefFn = make_function_t<int &&, Size>;
    RRefFn rref_fn = &test_apply_fn;

    // {
    //     A a{};
    //     CEXA_EXPECT(cexa::apply(val_fn, a));
    //     CEXA_EXPECT(cexa::apply(ref_fn, a));
    //     CEXA_EXPECT(cexa::apply(cref_fn, a));
    //     CEXA_EXPECT(cexa::apply(rref_fn, std::move(a)));
    // }
    // {
    //     ConstA a{};
    //     CEXA_EXPECT(cexa::apply(val_fn, a));
    //     CEXA_EXPECT(cexa::apply(cref_fn, a));
    // }
    {
        Tuple a{};
        CEXA_EXPECT(cexa::apply(val_fn, a));
        CEXA_EXPECT(cexa::apply(ref_fn, a));
        CEXA_EXPECT(cexa::apply(cref_fn, a));
        CEXA_EXPECT(cexa::apply(rref_fn, std::move(a)));
    }
    {
        CTuple a{};
        CEXA_EXPECT(cexa::apply(val_fn, a));
        CEXA_EXPECT(cexa::apply(cref_fn, a));
    }

}


template <std::size_t Size>
KOKKOS_INLINE_FUNCTION void test_one()
{
    // using A = std::array<int, Size>;
    using Tuple = make_tuple_t<int, Size>;

    using ValFn  = make_function_t<int, Size>;
    ValFn val_fn = &test_apply_fn;

    // {
    //     A a{};
    //     CEXA_EXPECT(cexa::apply(val_fn, a));
    // }
    {
        Tuple a{};
        CEXA_EXPECT(cexa::apply(val_fn, a));
    }
}

// clang-format off
CEXA_TEST(tuple_apply, apply_large_arity, (
    // Instantiate with 1-5 arguments.
    test_all<1>();
    test_all<2>();
    test_all<3>();
    test_all<4>();
    test_all<5>();

    // Stress test with 256
    // FIXME: compiling with 256 exceeds the max recursion depth on nvcc 12.2
#if !defined(KOKKOS_COMPILER_NVCC)
    test_one<256>();
#else
    test_one<64>();
#endif
))
// clang-format on
