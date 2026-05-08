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

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&) noexcept(see below) // noexcept since C++23

// Test with different ref/ptr/cv qualified argument types.

#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/type_id.h>

KOKKOS_INLINE_FUNCTION constexpr int constexpr_sum_fn() { return 0; }

template <class ...Ints>
KOKKOS_INLINE_FUNCTION constexpr int constexpr_sum_fn(int x1, Ints... rest) { return x1 + constexpr_sum_fn(rest...); }

struct ConstexprSumT {
  KOKKOS_DEFAULTED_FUNCTION constexpr ConstexprSumT() = default;
  template <class ...Ints>
  KOKKOS_INLINE_FUNCTION constexpr int operator()(Ints... values) const {
      return constexpr_sum_fn(values...);
  }
};


KOKKOS_INLINE_FUNCTION void test_constexpr_evaluation()
{
    constexpr ConstexprSumT sum_obj{};
    {
        using Tup = cexa::tuple<>;
        using Fn = int(&)();
        [[maybe_unused]] constexpr Tup t;
        static_assert(cexa::apply(static_cast<Fn>(constexpr_sum_fn), t) == 0, "");
        static_assert(cexa::apply(sum_obj, t) == 0, "");
    }
    {
        using Tup = cexa::tuple<int>;
        using Fn = int(&)(int);
        [[maybe_unused]] constexpr Tup t(42);
        static_assert(cexa::apply(static_cast<Fn>(constexpr_sum_fn), t) == 42, "");
        static_assert(cexa::apply(sum_obj, t) == 42, "");
    }
    {
        using Tup = cexa::tuple<int, long>;
        using Fn = int(&)(int, int);
        [[maybe_unused]] constexpr Tup t(42, 101);
        static_assert(cexa::apply(static_cast<Fn>(constexpr_sum_fn), t) == 143, "");
        static_assert(cexa::apply(sum_obj, t) == 143, "");
    }
    // {
    //     using Tup = std::pair<int, long>;
    //     using Fn = int(&)(int, int);
    //     constexpr Tup t(42, 101);
    //     static_assert(cexa::apply(static_cast<Fn>(constexpr_sum_fn), t) == 143, "");
    //     static_assert(cexa::apply(sum_obj, t) == 143, "");
    // }
    {
        using Tup = cexa::tuple<int, long, int>;
        using Fn = int(&)(int, int, int);
        [[maybe_unused]] constexpr Tup t(42, 101, -1);
        static_assert(cexa::apply(static_cast<Fn>(constexpr_sum_fn), t) == 142, "");
        static_assert(cexa::apply(sum_obj, t) == 142, "");
    }
    // {
    //     using Tup = std::array<int, 3>;
    //     using Fn = int(&)(int, int, int);
    //     constexpr Tup t = {42, 101, -1};
    //     static_assert(cexa::apply(static_cast<Fn>(constexpr_sum_fn), t) == 142, "");
    //     static_assert(cexa::apply(sum_obj, t) == 142, "");
    // }
}


enum CallQuals {
  CQ_None,
  CQ_LValue,
  CQ_ConstLValue,
  CQ_RValue,
  CQ_ConstRValue
};

template <class Tuple>
struct CallInfo {
  CallQuals quals;
  TypeID const* arg_types;
  Tuple args;

  template <class ...Args>
  CallInfo(CallQuals q, Args&&... xargs)
      : quals(q), arg_types(&makeArgumentID<Args&&...>()), args(std::forward<Args>(xargs)...)
  {}
};

template <class ...Args>
CallInfo<decltype(cexa::forward_as_tuple(std::declval<Args>()...))>
makeCallInfo(CallQuals quals, Args&&... args) {
    return {quals, std::forward<Args>(args)...};
}

struct TrackedCallable {

  TrackedCallable() = default;

  template <class ...Args> auto operator()(Args&&... xargs) &
  { return makeCallInfo(CQ_LValue, std::forward<Args>(xargs)...); }

  template <class ...Args> auto operator()(Args&&... xargs) const&
  { return makeCallInfo(CQ_ConstLValue, std::forward<Args>(xargs)...); }

  template <class ...Args> auto operator()(Args&&... xargs) &&
  { return makeCallInfo(CQ_RValue, std::forward<Args>(xargs)...); }

  template <class ...Args> auto operator()(Args&&... xargs) const&&
  { return makeCallInfo(CQ_ConstRValue, std::forward<Args>(xargs)...); }
};

template <class ...ExpectArgs, class Tuple>
void check_apply_quals_and_types(Tuple&& t) {
    TypeID const* const expect_args = &makeArgumentID<ExpectArgs...>();
    TrackedCallable obj;
    TrackedCallable const& cobj = obj;
    {
        auto ret = cexa::apply(obj, std::forward<Tuple>(t));
        CEXA_EXPECT_EQ(ret.quals, CQ_LValue);
        CEXA_EXPECT_EQ(ret.arg_types, expect_args);
        CEXA_EXPECT_EQ(ret.args, t);
    }
    {
        auto ret = cexa::apply(cobj, std::forward<Tuple>(t));
        CEXA_EXPECT_EQ(ret.quals, CQ_ConstLValue);
        CEXA_EXPECT_EQ(ret.arg_types, expect_args);
        CEXA_EXPECT_EQ(ret.args, t);
    }
    {
        auto ret = cexa::apply(std::move(obj), std::forward<Tuple>(t));
        CEXA_EXPECT_EQ(ret.quals, CQ_RValue);
        CEXA_EXPECT_EQ(ret.arg_types, expect_args);
        CEXA_EXPECT_EQ(ret.args, t);
    }
    {
        auto ret = cexa::apply(std::move(cobj), std::forward<Tuple>(t));
        CEXA_EXPECT_EQ(ret.quals, CQ_ConstRValue);
        CEXA_EXPECT_EQ(ret.arg_types, expect_args);
        CEXA_EXPECT_EQ(ret.args, t);
    }
}

void test_call_quals_and_arg_types()
{
    using Tup = cexa::tuple<int, int const&, unsigned&&>;
    const int x = 42;
    unsigned y = 101;
    Tup t(-1, x, std::move(y));
    Tup const& ct = t;
    check_apply_quals_and_types<int&, int const&, unsigned&>(t);
    check_apply_quals_and_types<int const&, int const&, unsigned&>(ct);
    check_apply_quals_and_types<int&&, int const&, unsigned&&>(std::move(t));
    check_apply_quals_and_types<int const&&, int const&, unsigned&&>(std::move(ct));
}


struct NothrowMoveable {
  KOKKOS_DEFAULTED_FUNCTION NothrowMoveable() noexcept = default;
  KOKKOS_INLINE_FUNCTION NothrowMoveable(NothrowMoveable const&) noexcept(false) {}
  KOKKOS_INLINE_FUNCTION NothrowMoveable(NothrowMoveable&&) noexcept {}
};

template <bool IsNoexcept>
struct TestNoexceptCallable {
  template <class ...Args>
  KOKKOS_INLINE_FUNCTION NothrowMoveable operator()(Args...) const noexcept(IsNoexcept) { return {}; }
};

KOKKOS_INLINE_FUNCTION void test_noexcept()
{
    TestNoexceptCallable<true> nec;
    TestNoexceptCallable<false> tc;
    {
        // test that the functions noexcept-ness is propagated
        using Tup = cexa::tuple<int, const char*, long>;
        [[maybe_unused]] Tup t;
#if TEST_STD_VER >= 23
        ASSERT_NOEXCEPT(cexa::apply(nec, t));
#endif
        ASSERT_NOT_NOEXCEPT(cexa::apply(tc, t));
    }
    {
        // test that the noexcept-ness of the argument conversions is checked.
        using Tup = cexa::tuple<NothrowMoveable, int>;
        [[maybe_unused]] Tup t;
        ASSERT_NOT_NOEXCEPT(cexa::apply(nec, t));
#if TEST_STD_VER >= 23
        ASSERT_NOEXCEPT(cexa::apply(nec, std::move(t)));
#endif
    }
}

namespace ReturnTypeTest {
#if defined(CEXA_ON_DEVICE)
    __device__ int my_int = 42;
#else
    int my_int = 42;
#endif

    template <int N> struct index {};

    KOKKOS_INLINE_FUNCTION void f(index<0>) {}

    KOKKOS_INLINE_FUNCTION int f(index<1>) { return 0; }

    KOKKOS_INLINE_FUNCTION int & f(index<2>) { return static_cast<int &>(my_int); }
    KOKKOS_INLINE_FUNCTION int const & f(index<3>) { return static_cast<int const &>(my_int); }
    KOKKOS_INLINE_FUNCTION int volatile & f(index<4>) { return static_cast<int volatile &>(my_int); }
    KOKKOS_INLINE_FUNCTION int const volatile & f(index<5>) { return static_cast<int const volatile &>(my_int); }

    KOKKOS_INLINE_FUNCTION int && f(index<6>) { return static_cast<int &&>(my_int); }
    KOKKOS_INLINE_FUNCTION int const && f(index<7>) { return static_cast<int const &&>(my_int); }
    KOKKOS_INLINE_FUNCTION int volatile && f(index<8>) { return static_cast<int volatile &&>(my_int); }
    KOKKOS_INLINE_FUNCTION int const volatile && f(index<9>) { return static_cast<int const volatile &&>(my_int); }

    KOKKOS_INLINE_FUNCTION int * f(index<10>) { return static_cast<int *>(&my_int); }
    KOKKOS_INLINE_FUNCTION int const * f(index<11>) { return static_cast<int const *>(&my_int); }
    KOKKOS_INLINE_FUNCTION int volatile * f(index<12>) { return static_cast<int volatile *>(&my_int); }
    KOKKOS_INLINE_FUNCTION int const volatile * f(index<13>) { return static_cast<int const volatile *>(&my_int); }

    template <int Func, class Expect>
    KOKKOS_INLINE_FUNCTION void test()
    {
        using RawInvokeResult = decltype(f(index<Func>{}));
        static_assert(std::is_same<RawInvokeResult, Expect>::value, "");
        using FnType = RawInvokeResult (*) (index<Func>);
        [[maybe_unused]] FnType fn = f;
        [[maybe_unused]] cexa::tuple<index<Func>> t;
        using InvokeResult = decltype(cexa::apply(fn, t));
        static_assert(std::is_same<InvokeResult, Expect>::value, "");
    }
} // namespace ReturnTypeTest

KOKKOS_INLINE_FUNCTION void test_return_type()
{
    using ReturnTypeTest::test;
    test<0, void>();
    test<1, int>();
    test<2, int &>();
    test<3, int const &>();
    test<4, int volatile &>();
    test<5, int const volatile &>();
    test<6, int &&>();
    test<7, int const &&>();
    test<8, int volatile &&>();
    test<9, int const volatile &&>();
    test<10, int *>();
    test<11, int const *>();
    test<12, int volatile *>();
    test<13, int const volatile *>();
}

TEST(host_tuple_apply, apply_host) {
    test_call_quals_and_arg_types();
}

// clang-format off
CEXA_TEST(tuple_apply, apply, (
    test_constexpr_evaluation();
    test_return_type();
    test_noexcept();
))
// clang-format on
