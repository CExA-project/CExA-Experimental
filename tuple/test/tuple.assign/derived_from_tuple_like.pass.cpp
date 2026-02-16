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

// template <class... UTypes>
//   tuple& operator=(const tuple<UTypes...>& u);

// UNSUPPORTED: c++03

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <support/propagate_value_category.hpp>

struct TracksIntQuals {
  KOKKOS_INLINE_FUNCTION TracksIntQuals() : value(-1), value_category(VC_None), assigned(false) {}

  template <class Tp,
            class = typename std::enable_if<!std::is_same<
                typename std::decay<Tp>::type, TracksIntQuals>::value>::type>
  KOKKOS_INLINE_FUNCTION TracksIntQuals(Tp &&x)
      : value(x), value_category(getValueCategory<Tp &&>()), assigned(false) {
    static_assert(std::is_same<UnCVRef<Tp>, int>::value, "");
  }

  template <class Tp,
            class = typename std::enable_if<!std::is_same<
                typename std::decay<Tp>::type, TracksIntQuals>::value>::type>
  KOKKOS_INLINE_FUNCTION TracksIntQuals &operator=(Tp &&x) {
    static_assert(std::is_same<UnCVRef<Tp>, int>::value, "");
    value = x;
    value_category = getValueCategory<Tp &&>();
    assigned = true;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION void reset() {
    value = -1;
    value_category = VC_None;
    assigned = false;
  }

  KOKKOS_INLINE_FUNCTION bool checkConstruct(int expect, ValueCategory expect_vc) const {
    return value != 1 && value == expect && value_category == expect_vc &&
           assigned == false;
  }

  KOKKOS_INLINE_FUNCTION bool checkAssign(int expect, ValueCategory expect_vc) const {
    return value != 1 && value == expect && value_category == expect_vc &&
           assigned == true;
  }

  int value;
  ValueCategory value_category;
  bool assigned;
};

template <class Tup>
struct DerivedFromTup : Tup {
  using Tup::Tup;
};

template <ValueCategory VC>
KOKKOS_INLINE_FUNCTION void do_derived_assign_test() {
  using Tup1 = cexa::tuple<long, TracksIntQuals>;
  Tup1 t;
  auto reset = [&]() {
    cexa::get<0>(t) = -1;
    cexa::get<1>(t).reset();
  };
  {
    DerivedFromTup<cexa::tuple<int, int>> d;
    cexa::get<0>(d) = 42;
    cexa::get<1>(d) = 101;

    t = ValueCategoryCast<VC>(d);
    CEXA_EXPECT_EQ(cexa::get<0>(t), 42);
    CEXA_EXPECT(cexa::get<1>(t).checkAssign(101, VC));
  }
  reset();
// FIXME: disabled because cexa::get<> doesn't work for types other than cexa::tuple
//   {
//     DerivedFromTup<std::pair<int, int>> d;
//     cexa::get<0>(d) = 42;
//     cexa::get<1>(d) = 101;
//
//     t = ValueCategoryCast<VC>(d);
//     CEXA_EXPECT_EQ(cexa::get<0>(t), 42);
//     CEXA_EXPECT(cexa::get<1>(t).checkAssign(101, VC));
//   }
//   reset();
//   {
// #ifdef _LIBCPP_VERSION // assignment from std::array is a libc++ extension
//     DerivedFromTup<std::array<int, 2>> d;
//     cexa::get<0>(d) = 42;
//     cexa::get<1>(d) = 101;
//
//     t = ValueCategoryCast<VC>(d);
//     CEXA_EXPECT_EQ(cexa::get<0>(t), 42);
//     CEXA_EXPECT(cexa::get<1>(t).checkAssign(101, VC));
// #endif
//   }
}

// clang-format on
CEXA_TEST(tuple_assign, derived_from_tuple_like, (
    do_derived_assign_test<VC_LVal | VC_Const>();
    do_derived_assign_test<VC_RVal>();
))
// clang-format off
