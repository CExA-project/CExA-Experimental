// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <sstream>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "util.hpp"

TEST(Assign_Fwd, SameType) {
  Cexa::Experimental::variant<int, std::string> v(101);
  EXPECT_EQ(101, Cexa::Experimental::get<int>(v));
  v = 202;
  EXPECT_EQ(202, Cexa::Experimental::get<int>(v));
}

TEST(Assign_Fwd, DiffType) {
  Cexa::Experimental::variant<int, std::string> v(42);
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
  v = "42";
  EXPECT_EQ("42", Cexa::Experimental::get<std::string>(v));
}

TEST(Assign_Fwd, ExactMatch) {
  Cexa::Experimental::variant<const char *, std::string> v;
  v = std::string("hello");
  EXPECT_EQ("hello", Cexa::Experimental::get<std::string>(v));
}

TEST(Assign_Fwd, BetterMatch) {
  Cexa::Experimental::variant<int, double> v;
  // `char` -> `int` is better than `char` -> `double`
  v = 'x';
  EXPECT_EQ(static_cast<int>('x'), Cexa::Experimental::get<int>(v));
}

TEST(Assign_Fwd, NoMatch) {
  struct x {};
  static_assert(
      !std::is_assignable<Cexa::Experimental::variant<int, std::string>, x>{},
      "variant<int, std::string> v; v = x;");
}

TEST(Assign_Fwd, WideningOrAmbiguous) {
#if defined(__clang__) || !defined(__GNUC__) || __GNUC__ >= 5
  static_assert(
      std::is_assignable<Cexa::Experimental::variant<short, long>, int>{},
      "variant<short, long> v; v = 42;");
#else
  static_assert(
      !std::is_assignable<Cexa::Experimental::variant<short, long>, int>{},
      "variant<short, long> v; v = 42;");
#endif
}

TEST(Assign_Fwd, SameTypeOptimization) {
  Cexa::Experimental::variant<int, std::string> v("hello world!");
  // Check `v`.
  const std::string &x = Cexa::Experimental::get<std::string>(v);
  EXPECT_EQ("hello world!", x);
  // Save the "hello world!"'s capacity.
  auto capacity = x.capacity();
  // Use `std::string::operator=(const char *)` to assign into `v`.
  v = "hello";
  // Check `v`.
  const std::string &y = Cexa::Experimental::get<std::string>(v);
  EXPECT_EQ("hello", y);
  // Since "hello" is shorter than "hello world!", we should have preserved the
  // existing capacity of the string!.
  EXPECT_EQ(capacity, y.capacity());
}

#ifdef MPARK_EXCEPTIONS
TEST(Assign_Fwd, ThrowOnAssignment) {
  Cexa::Experimental::variant<int, move_thrower_t> v(
      Cexa::Experimental::in_place_type_t<move_thrower_t>{});
  // Since `variant` is already in `move_thrower_t`, assignment optimization
  // kicks and we simply invoke
  // `move_thrower_t &operator=(move_thrower_t &&);` which throws.
  EXPECT_THROW(v = move_thrower_t{}, MoveAssignment);
  EXPECT_FALSE(v.valueless_by_exception());
  EXPECT_EQ(1u, v.index());
  // We can still assign into a variant in an invalid state.
  v = 42;
  // Check `v`.
  EXPECT_FALSE(v.valueless_by_exception());
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
}
#endif

#if 0
TEST(Assign_Fwd, ThrowOnTemporaryConstruction) {
  Cexa::Experimental::variant<int, copy_thrower_t> v(42);
  // Since `copy_thrower_t`'s copy constructor always throws, we will fail to
  // construct the variant. This results in our variant staying in
  // its original state.
  copy_thrower_t copy_thrower{};
  EXPECT_THROW(v = copy_thrower, CopyConstruction);
  EXPECT_FALSE(v.valueless_by_exception());
  EXPECT_EQ(0u, v.index());
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
}

TEST(Assign_Fwd, ThrowOnVariantConstruction) {
  Cexa::Experimental::variant<int, move_thrower_t> v(42);
  // Since `move_thrower_t`'s copy constructor never throws, we successfully
  // construct the temporary object by copying `move_thrower_t`. We then
  // proceed to move the temporary object into our variant, at which point
  // `move_thrower_t`'s move constructor throws. This results in our `variant`
  // transitioning into the invalid state.
  move_thrower_t move_thrower;
  EXPECT_THROW(v = move_thrower, MoveConstruction);
  EXPECT_TRUE(v.valueless_by_exception());
  // We can still assign into a variant in an invalid state.
  v = 42;
  // Check `v`.
  EXPECT_FALSE(v.valueless_by_exception());
  EXPECT_EQ(42, Cexa::Experimental::get<int>(v));
}
#endif
