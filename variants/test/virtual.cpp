
#include <Kokkos_Variant.hpp>

#include <gtest/gtest.h>

#include "util.hpp"

struct A {
  KOKKOS_FUNCTION virtual test_util::DeviceString func(int in) {
    return test_util::DeviceString("A") + in;
  }
};

struct B : public A {
  KOKKOS_FUNCTION test_util::DeviceString func(int in) {
    return A::func(in) + test_util::DeviceString("B") + in;
  }
};

struct C : public A {
  KOKKOS_FUNCTION test_util::DeviceString func(int in) {
    return test_util::DeviceString("C") + in;
  }
};

using FunctionVariant = cexa::experimental::variant<A, B, C>;

KOKKOS_FUNCTION auto call_func(const FunctionVariant &variant, int in) {
  return cexa::experimental::visit([in](auto a) { return a.func(in); },
                                   variant);
}

void test_helper_VirtualFct() {
  const int ntests = 10;
  FunctionVariant var;
  int errors;

  var = A{};
  Kokkos::parallel_reduce(
      "Test_A", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        test_util::DeviceString res      = call_func(var, i);
        test_util::DeviceString expected = test_util::DeviceString("A") + i;
        DEXPECT_EQ(res, expected);
      },
      errors);
  EXPECT_EQ(0, errors);

  var = B{};
  Kokkos::parallel_reduce(
      "Test_B", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        test_util::DeviceString res = call_func(var, i);
        test_util::DeviceString expected =
            test_util::DeviceString("A") + i + "B" + i;
        DEXPECT_EQ(res, expected);
      },
      errors);
  EXPECT_EQ(0, errors);

  var = C{};
  Kokkos::parallel_reduce(
      "Test_B", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        test_util::DeviceString res      = call_func(var, i);
        test_util::DeviceString expected = test_util::DeviceString("C") + i;
        DEXPECT_EQ(res, expected);
      },
      errors);
  EXPECT_EQ(0, errors);
}

TEST(Virtual, test1) { test_helper_VirtualFct(); }

TEST_MAIN
