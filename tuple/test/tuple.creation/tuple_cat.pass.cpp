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

// template <class... Tuples> tuple<CTypes...> tuple_cat(Tuples&&... tpls);

// UNSUPPORTED: c++03

#include <utility>
#include <array>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/MoveOnly.h>

namespace NS {
struct Namespaced {
  int i;
};
template<typename ...Ts>
void forward_as_tuple(Ts...) = delete;
}

// https://llvm.org/PR41689
struct Unconstrained {
  int data;
  template <typename Arg>
   KOKKOS_INLINE_FUNCTION constexpr Unconstrained(Arg arg) : data(arg) {}
};

KOKKOS_INLINE_FUNCTION constexpr bool test_tuple_cat_with_unconstrained_constructor() {
  {
    auto tup_src = cexa::tuple<Unconstrained>(Unconstrained(5));
    auto tup     = cexa::tuple_cat(tup_src);
    CEXA_EXPECT_EQ(cexa::get<0>(tup).data, 5);
  }
  {
    auto tup = cexa::tuple_cat(cexa::tuple<Unconstrained>(Unconstrained(6)));
    CEXA_EXPECT_EQ(cexa::get<0>(tup).data, 6);
  }
  {
    auto tup = cexa::tuple_cat(cexa::tuple<Unconstrained>(Unconstrained(7)), cexa::tuple<>());
    CEXA_EXPECT_EQ(cexa::get<0>(tup).data, 7);
  }
  {
    auto tup_src = cexa::tuple(Unconstrained(8));
    auto tup     = cexa::tuple_cat(tup_src);
    ASSERT_SAME_TYPE(decltype(tup), cexa::tuple<Unconstrained>);
    CEXA_EXPECT_EQ(cexa::get<0>(tup).data, 8);
  }
  {
    auto tup = cexa::tuple_cat(cexa::tuple(Unconstrained(9)));
    ASSERT_SAME_TYPE(decltype(tup), cexa::tuple<Unconstrained>);
    CEXA_EXPECT_EQ(cexa::get<0>(tup).data, 9);
  }
  {
    auto tup = cexa::tuple_cat(cexa::tuple(Unconstrained(10)), cexa::tuple());
    ASSERT_SAME_TYPE(decltype(tup), cexa::tuple<Unconstrained>);
    CEXA_EXPECT_EQ(cexa::get<0>(tup).data, 10);
  }
  return true;
}

TEST(host_tuple_creation, tuple_cat_host) {
    {
        cexa::tuple<> t = cexa::tuple_cat(std::array<int, 0>());
        ((void)t); // Prevent unused warning
    }
    {
        constexpr cexa::tuple<> t = cexa::tuple_cat(std::array<int, 0>());
        ((void)t); // Prevent unused warning
    }
    {
        cexa::tuple<int, int, int> t = cexa::tuple_cat(std::array<int, 3>());
        CEXA_EXPECT_EQ(cexa::get<0>(t), 0);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 0);
        CEXA_EXPECT_EQ(cexa::get<2>(t), 0);
    }
    {
        cexa::tuple<int, MoveOnly> t = cexa::tuple_cat(std::pair<int, MoveOnly>(2, 1));
        CEXA_EXPECT_EQ(cexa::get<0>(t), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 1);
    }
}

// clang-format off
CEXA_TEST(tuple_creation, tuple_cat, (
    {
        [[maybe_unused]] cexa::tuple<> t = cexa::tuple_cat();
    }
    {
        cexa::tuple<> t1;
        [[maybe_unused]] cexa::tuple<> t2 = cexa::tuple_cat(t1);
    }
    {
        [[maybe_unused]] cexa::tuple<> t = cexa::tuple_cat(cexa::tuple<>());
    }
    {
        cexa::tuple<int> t1(1);
        cexa::tuple<int> t = cexa::tuple_cat(t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
    }

    {
        [[maybe_unused]] constexpr cexa::tuple<> t = cexa::tuple_cat();
    }
    {
        constexpr cexa::tuple<> t1;
        [[maybe_unused]] constexpr cexa::tuple<> t2 = cexa::tuple_cat(t1);
    }
    {
        [[maybe_unused]] constexpr cexa::tuple<> t = cexa::tuple_cat(cexa::tuple<>());
    }
    {
        constexpr cexa::tuple<int> t1(1);
        constexpr cexa::tuple<int> t = cexa::tuple_cat(t1);
        static_assert(cexa::get<0>(t) == 1, "");
    }
    {
        constexpr cexa::tuple<int> t1(1);
        constexpr cexa::tuple<int, int> t = cexa::tuple_cat(t1, t1);
        static_assert(cexa::get<0>(t) == 1, "");
        static_assert(cexa::get<1>(t) == 1, "");
    }
    {
        cexa::tuple<int, MoveOnly> t =
                                cexa::tuple_cat(cexa::tuple<int, MoveOnly>(1, 2));
        CEXA_EXPECT_EQ(cexa::get<0>(t), 1);
        CEXA_EXPECT_EQ(cexa::get<1>(t), 2);
    }

    {
        cexa::tuple<> t1;
        cexa::tuple<> t2;
        [[maybe_unused]] cexa::tuple<> t3 = cexa::tuple_cat(t1, t2);
    }
    {
        cexa::tuple<> t1;
        cexa::tuple<int> t2(2);
        cexa::tuple<int> t3 = cexa::tuple_cat(t1, t2);
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 2);
    }
    {
        cexa::tuple<> t1;
        cexa::tuple<int> t2(2);
        cexa::tuple<int> t3 = cexa::tuple_cat(t2, t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 2);
    }
    {
        cexa::tuple<int*> t1;
        cexa::tuple<int> t2(2);
        cexa::tuple<int*, int> t3 = cexa::tuple_cat(t1, t2);
        CEXA_EXPECT_EQ(cexa::get<0>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 2);
    }
    {
        cexa::tuple<int*> t1;
        cexa::tuple<int> t2(2);
        cexa::tuple<int, int*> t3 = cexa::tuple_cat(t2, t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), nullptr);
    }
    {
        cexa::tuple<int*> t1;
        cexa::tuple<int, double> t2(2, 3.5);
        cexa::tuple<int*, int, double> t3 = cexa::tuple_cat(t1, t2);
        CEXA_EXPECT_EQ(cexa::get<0>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), 3.5);
    }
    {
        cexa::tuple<int*> t1;
        cexa::tuple<int, double> t2(2, 3.5);
        cexa::tuple<int, double, int*> t3 = cexa::tuple_cat(t2, t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 3.5);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), nullptr);
    }
    {
        cexa::tuple<int*, MoveOnly> t1(nullptr, 1);
        cexa::tuple<int, double> t2(2, 3.5);
        cexa::tuple<int*, MoveOnly, int, double> t3 =
                                              cexa::tuple_cat(std::move(t1), t2);
        CEXA_EXPECT_EQ(cexa::get<0>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 1);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<3>(t3), 3.5);
    }
    {
        cexa::tuple<int*, MoveOnly> t1(nullptr, 1);
        cexa::tuple<int, double> t2(2, 3.5);
        cexa::tuple<int, double, int*, MoveOnly> t3 =
                                              cexa::tuple_cat(t2, std::move(t1));
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 3.5);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<3>(t3), 1);
    }
    {
        cexa::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cexa::tuple<int*, MoveOnly> t2(nullptr, 4);
        cexa::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   cexa::tuple_cat(std::move(t1), std::move(t2));
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 1);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<3>(t3), 4);
    }

    {
        cexa::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cexa::tuple<int*, MoveOnly> t2(nullptr, 4);
        cexa::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   cexa::tuple_cat(cexa::tuple<>(),
                                                  std::move(t1),
                                                  std::move(t2));
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 1);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<3>(t3), 4);
    }
    {
        cexa::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cexa::tuple<int*, MoveOnly> t2(nullptr, 4);
        cexa::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   cexa::tuple_cat(std::move(t1),
                                                  cexa::tuple<>(),
                                                  std::move(t2));
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 1);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<3>(t3), 4);
    }
    {
        cexa::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cexa::tuple<int*, MoveOnly> t2(nullptr, 4);
        cexa::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   cexa::tuple_cat(std::move(t1),
                                                  std::move(t2),
                                                  cexa::tuple<>());
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 1);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<3>(t3), 4);
    }
    {
        cexa::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cexa::tuple<int*, MoveOnly> t2(nullptr, 4);
        cexa::tuple<MoveOnly, MoveOnly, int*, MoveOnly, int> t3 =
                                   cexa::tuple_cat(std::move(t1),
                                                  std::move(t2),
                                                  cexa::tuple<int>(5));
        CEXA_EXPECT_EQ(cexa::get<0>(t3), 1);
        CEXA_EXPECT_EQ(cexa::get<1>(t3), 2);
        CEXA_EXPECT_EQ(cexa::get<2>(t3), nullptr);
        CEXA_EXPECT_EQ(cexa::get<3>(t3), 4);
        CEXA_EXPECT_EQ(cexa::get<4>(t3), 5);
    }
    {
        // See bug #19616.
        auto t1 = cexa::tuple_cat(
            cexa::make_tuple(cexa::make_tuple(1)),
            cexa::make_tuple()
        );
        CEXA_EXPECT_EQ(t1, cexa::make_tuple(cexa::make_tuple(1)));

        auto t2 = cexa::tuple_cat(
            cexa::make_tuple(cexa::make_tuple(1)),
            cexa::make_tuple(cexa::make_tuple(2))
        );
        CEXA_EXPECT_EQ(t2, cexa::make_tuple(cexa::make_tuple(1), cexa::make_tuple(2)));
    }
    {
        int x = 101;
        cexa::tuple<int, const int, int&, const int&, int&&> t(42, 101, x, x, std::move(x));
        const auto& ct = t;
        cexa::tuple<int, const int, int&, const int&> t2(42, 101, x, x);
        const auto& ct2 = t2;

        auto r = cexa::tuple_cat(std::move(t), std::move(ct), t2, ct2);

        ASSERT_SAME_TYPE(decltype(r), cexa::tuple<
            int, const int, int&, const int&, int&&,
            int, const int, int&, const int&, int&&,
            int, const int, int&, const int&,
            int, const int, int&, const int&>);
        ((void)r);
    }
    {
        cexa::tuple<NS::Namespaced> t1(NS::Namespaced{1});
        cexa::tuple<NS::Namespaced> t = cexa::tuple_cat(t1);
        cexa::tuple<NS::Namespaced, NS::Namespaced> t2 =
            cexa::tuple_cat(t1, t1);
        CEXA_EXPECT_EQ(cexa::get<0>(t).i, 1);
        CEXA_EXPECT_EQ(cexa::get<0>(t2).i, 1);
    }
    // See https://llvm.org/PR41689
    {
      test_tuple_cat_with_unconstrained_constructor();
      static_assert(test_tuple_cat_with_unconstrained_constructor(), "");
    }
))
// clang-format on
