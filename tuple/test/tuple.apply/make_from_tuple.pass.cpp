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

// template <class T, class Tuple> constexpr T make_from_tuple(Tuple&&);

#include <cstdint>
#include <utility>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/type_id.h>

template <class Tuple>
struct ConstexprConstructibleFromTuple {
  template <class ...Args>
  explicit constexpr ConstexprConstructibleFromTuple(Args&&... xargs)
      : args{std::forward<Args>(xargs)...} {}
  Tuple args;
};

template <class TupleLike>
struct ConstructibleFromTuple;

template <template <class ...> class Tuple, class ...Types>
struct ConstructibleFromTuple<Tuple<Types...>> {
  template <class ...Args>
  explicit ConstructibleFromTuple(Args&&... xargs)
      : args(xargs...),
        arg_types(&makeArgumentID<Args&&...>())
  {}
  Tuple<std::decay_t<Types>...> args;
  TypeID const* arg_types;
};

// template <class Tp, std::size_t N>
// struct ConstructibleFromTuple<std::array<Tp, N>> {
// template <class ...Args>
//   explicit ConstructibleFromTuple(Args&&... xargs)
//       : args{xargs...},
//         arg_types(&makeArgumentID<Args&&...>())
//   {}
//   std::array<Tp, N> args;
//   TypeID const* arg_types;
// };

template <class Tuple>
constexpr bool do_constexpr_test(Tuple&& tup) {
    using RawTuple = std::decay_t<Tuple>;
    using Tp = ConstexprConstructibleFromTuple<RawTuple>;
    return cexa::make_from_tuple<Tp>(std::forward<Tuple>(tup)).args == tup;
}

template <class ...ExpectTypes, class Tuple>
bool do_forwarding_test(Tuple&& tup) {
    using RawTuple = std::decay_t<Tuple>;
    using Tp = ConstructibleFromTuple<RawTuple>;
    const Tp value = cexa::make_from_tuple<Tp>(std::forward<Tuple>(tup));
    return value.args == tup
        && value.arg_types == &makeArgumentID<ExpectTypes...>();
}

constexpr bool test_constexpr_construction() {
    {
        constexpr cexa::tuple<> tup;
        static_assert(do_constexpr_test(tup), "");
    }
    {
        constexpr cexa::tuple<int> tup(42);
        static_assert(do_constexpr_test(tup), "");
    }
    {
        constexpr cexa::tuple<int, long, void*> tup(42, 101, nullptr);
        static_assert(do_constexpr_test(tup), "");
    }
    // {
    //     constexpr std::pair<int, const char*> p(42, "hello world");
    //     static_assert(do_constexpr_test(p), "");
    // }
    // {
    //     using Tuple = std::array<int, 3>;
    //     using ValueTp = ConstexprConstructibleFromTuple<Tuple>;
    //     constexpr Tuple arr = {42, 101, -1};
    //     constexpr ValueTp value = cexa::make_from_tuple<ValueTp>(arr);
    //     static_assert(value.args[0] == arr[0] && value.args[1] == arr[1]
    //         && value.args[2] == arr[2], "");
    // }
    return true;
}

void test_perfect_forwarding() {
    {
        using Tup = cexa::tuple<>;
        Tup tup;
        Tup const& ctup = tup;
        CEXA_EXPECT(do_forwarding_test<>(tup));
        CEXA_EXPECT(do_forwarding_test<>(ctup));
    }
    {
        using Tup = cexa::tuple<int>;
        Tup tup(42);
        Tup const& ctup = tup;
        CEXA_EXPECT(do_forwarding_test<int&>(tup));
        CEXA_EXPECT(do_forwarding_test<int const&>(ctup));
        CEXA_EXPECT(do_forwarding_test<int&&>(std::move(tup)));
        CEXA_EXPECT(do_forwarding_test<int const&&>(std::move(ctup)));
    }
    {
        using Tup = cexa::tuple<int&, const char*, unsigned&&>;
        int x = 42;
        unsigned y = 101;
        Tup tup(x, "hello world", std::move(y));
        Tup const& ctup = tup;
        CEXA_EXPECT((do_forwarding_test<int&, const char*&, unsigned&>(tup)));
        CEXA_EXPECT((do_forwarding_test<int&, const char* const&, unsigned &>(ctup)));
        CEXA_EXPECT((do_forwarding_test<int&, const char*&&, unsigned&&>(std::move(tup))));
        CEXA_EXPECT((do_forwarding_test<int&, const char* const&&, unsigned &&>(std::move(ctup))));
    }
    // test with pair<T, U>
    // {
    //     using Tup = std::pair<int&, const char*>;
    //     int x = 42;
    //     Tup tup(x, "hello world");
    //     Tup const& ctup = tup;
    //     CEXA_EXPECT((do_forwarding_test<int&, const char*&>(tup)));
    //     CEXA_EXPECT((do_forwarding_test<int&, const char* const&>(ctup)));
    //     CEXA_EXPECT((do_forwarding_test<int&, const char*&&>(std::move(tup))));
    //     CEXA_EXPECT((do_forwarding_test<int&, const char* const&&>(std::move(ctup))));
    // }
    // test with array<T, I>
    // {
    //     using Tup = std::array<int, 3>;
    //     Tup tup = {42, 101, -1};
    //     Tup const& ctup = tup;
    //     CEXA_EXPECT((do_forwarding_test<int&, int&, int&>(tup)));
    //     CEXA_EXPECT((do_forwarding_test<int const&, int const&, int const&>(ctup)));
    //     CEXA_EXPECT((do_forwarding_test<int&&, int&&, int&&>(std::move(tup))));
    //     CEXA_EXPECT((do_forwarding_test<int const&&, int const&&, int const&&>(std::move(ctup))));
    // }
}

// FIXME: add this later
// void test_noexcept() {
//     struct NothrowMoveable {
//       NothrowMoveable() = default;
//       NothrowMoveable(NothrowMoveable const&) {}
//       NothrowMoveable(NothrowMoveable&&) noexcept {}
//     };
//     struct TestType {
//       TestType(int, NothrowMoveable) noexcept {}
//       TestType(int, int, int) noexcept(false) {}
//       TestType(long, long, long) noexcept {}
//     };
//     {
//         using Tuple = cexa::tuple<int, NothrowMoveable>;
//         Tuple tup; ((void)tup);
//         Tuple const& ctup = tup; ((void)ctup);
//         ASSERT_NOT_NOEXCEPT(cexa::make_from_tuple<TestType>(ctup));
//         LIBCPP_ASSERT_NOEXCEPT(cexa::make_from_tuple<TestType>(std::move(tup)));
//     }
//     {
//         using Tuple = std::pair<int, NothrowMoveable>;
//         Tuple tup; ((void)tup);
//         Tuple const& ctup = tup; ((void)ctup);
//         ASSERT_NOT_NOEXCEPT(cexa::make_from_tuple<TestType>(ctup));
//         LIBCPP_ASSERT_NOEXCEPT(cexa::make_from_tuple<TestType>(std::move(tup)));
//     }
//     {
//         using Tuple = cexa::tuple<int, int, int>;
//         Tuple tup; ((void)tup);
//         ASSERT_NOT_NOEXCEPT(cexa::make_from_tuple<TestType>(tup));
//     }
//     {
//         using Tuple = cexa::tuple<long, long, long>;
//         Tuple tup; ((void)tup);
//         LIBCPP_ASSERT_NOEXCEPT(cexa::make_from_tuple<TestType>(tup));
//     }
//     {
//         using Tuple = std::array<int, 3>;
//         Tuple tup; ((void)tup);
//         ASSERT_NOT_NOEXCEPT(cexa::make_from_tuple<TestType>(tup));
//     }
//     {
//         using Tuple = std::array<long, 3>;
//         Tuple tup; ((void)tup);
//         LIBCPP_ASSERT_NOEXCEPT(cexa::make_from_tuple<TestType>(tup));
//     }
// }

namespace LWG3528 {
template <class T, class Tuple>
auto test_make_from_tuple(T&&, Tuple&& t) -> decltype(cexa::make_from_tuple<T>(t), std::uint8_t()) {
  return 0;
}
template <class T, class Tuple>
uint32_t test_make_from_tuple(...) {
  return 0;
}

template <class T, class Tuple>
static constexpr bool can_make_from_tuple =
    std::is_same_v<decltype(test_make_from_tuple<T, Tuple>(T{}, Tuple{})), std::uint8_t>;

struct A {
  int a;
};
struct B : public A {};

struct C {
  C(const B&) {}
};

enum class D {
  ONE,
  TWO,
};

// Test cexa::make_from_tuple constraints.

// reinterpret_cast
static_assert(!can_make_from_tuple<int*, cexa::tuple<A*>>);
static_assert(can_make_from_tuple<A*, cexa::tuple<A*>>);

// const_cast
static_assert(!can_make_from_tuple<char*, cexa::tuple<const char*>>);
static_assert(!can_make_from_tuple<volatile char*, cexa::tuple<const volatile char*>>);
static_assert(can_make_from_tuple<volatile char*, cexa::tuple<volatile char*>>);
static_assert(can_make_from_tuple<char*, cexa::tuple<char*>>);
static_assert(can_make_from_tuple<const char*, cexa::tuple<char*>>);
static_assert(can_make_from_tuple<const volatile char*, cexa::tuple<volatile char*>>);

// static_cast
static_assert(!can_make_from_tuple<int, cexa::tuple<D>>);
static_assert(!can_make_from_tuple<D, cexa::tuple<int>>);
static_assert(can_make_from_tuple<long, cexa::tuple<int>>);
static_assert(can_make_from_tuple<double, cexa::tuple<float>>);
static_assert(can_make_from_tuple<float, cexa::tuple<double>>);

} // namespace LWG3528

static_assert(LWG3528::can_make_from_tuple<int, cexa::tuple<>>);

static_assert(test_constexpr_construction());

struct Empty {};
struct ThreeArgs {
    int i;
    float f;
    unsigned u;
    KOKKOS_INLINE_FUNCTION ThreeArgs(int i, float f, unsigned u) : i(i), f(f), u(u) {}
};

TEST(host_tuple_apply, make_from_tuple_host) {
    test_perfect_forwarding();
    // test_noexcept();
}

// clang-format off
CEXA_TEST(tuple_apply, make_from_tuple, (
    {
        using Tup = cexa::tuple<>;
        Tup tup;
        Tup const& ctup = tup;
        [[maybe_unused]] Empty e = cexa::make_from_tuple<Empty>(tup);
        [[maybe_unused]] Empty ce = cexa::make_from_tuple<Empty>(ctup);
    }
    {
        using Tup = cexa::tuple<int>;
        Tup tup(42);
        Tup const& ctup = tup;
        int& iref = cexa::make_from_tuple<int&>(tup);
        CEXA_EXPECT_EQ(iref, 42);
        iref = 51;
        CEXA_EXPECT_EQ(cexa::get<0>(tup), 51);
        int const& icref = cexa::make_from_tuple<int const&>(ctup);
        CEXA_EXPECT_EQ(icref, 51);
    }
    {
        using Tup = cexa::tuple<int, float, unsigned>;
        int i = 42;
        float f = 1.f;
        unsigned u = 101;
        Tup tup(i, f, u);
        ThreeArgs t = cexa::make_from_tuple<ThreeArgs>(tup);
        CEXA_EXPECT_EQ(t.i, i);
        CEXA_EXPECT_EQ(t.f, f);
        CEXA_EXPECT_EQ(t.u, u);
    }
))
// clang-format on
