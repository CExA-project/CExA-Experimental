
#include <Kokkos_Variant.hpp>

#include <gtest/gtest.h>

#include "util.hpp"

struct Function_Add {
  template <typename Real> KOKKOS_FUNCTION Real operator()(Real in) {
    return in + in;
  }

  template <typename Real> KOKKOS_FUNCTION Real operator()(Real in1, Real in2) {
    return in1 + in2;
  }
};

struct Function_Mul {
  template <typename Real> KOKKOS_FUNCTION Real operator()(Real in) {
    return in * in;
  }

  template <typename Real> KOKKOS_FUNCTION Real operator()(Real in1, Real in2) {
    return in1 * in2;
  }
};

struct Function_Pow {
  template <typename Real> KOKKOS_FUNCTION Real operator()(Real in) {
    return Kokkos::pow(in, in);
  }

  template <typename Real> KOKKOS_FUNCTION Real operator()(Real in1, Real in2) {
    return Kokkos::pow(in1, in2);
  }
};

using FunctionVariant =
    Cexa::Experimental::variant<Function_Add, Function_Mul,
                                Function_Pow>;

template <typename Variant, typename... Args>
KOKKOS_FUNCTION auto call_variant(const Variant& variant, const Args&... args) {
  return Cexa::Experimental::visit([args...](auto f) { return f(args...); },
                                   variant);
}

template <typename Real> void test_helper_Function_Ptr() {
  const int ntests = 10;
  FunctionVariant fvariant;
  int errors;

  fvariant = Function_Add{};
  Kokkos::parallel_reduce(
      "Test_Add1", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        Real a = (Real)i;
        Real res = call_variant(fvariant, a);
        DEXPECT_EQ(res, a + a);
      },
      errors);

  EXPECT_EQ(0, errors);

  Kokkos::parallel_reduce(
      "Test_Add2", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        Real a = (Real)i;
        Real b = (Real)2;
        Real res = call_variant(fvariant, a, b);
        DEXPECT_EQ(res, a + b);
      },
      errors);
  EXPECT_EQ(0, errors);

  fvariant = Function_Mul{};
  Kokkos::parallel_reduce(
      "Test_Mul1", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        Real a = (Real)i;
        Real res = call_variant(fvariant, a);
        DEXPECT_EQ(res, a * a);
      },
      errors);
  EXPECT_EQ(0, errors);

  Kokkos::parallel_reduce(
      "Test_Mul2", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        Real a = (Real)i;
        Real b = (Real)2;
        Real res = call_variant(fvariant, a, b);
        DEXPECT_EQ(res, a * b);
      },
      errors);
  EXPECT_EQ(0, errors);

  fvariant = Function_Pow{};
  Kokkos::parallel_reduce(
      "Test_Pow1", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        Real a = (Real)i;
        Real res = call_variant(fvariant, a);
        DEXPECT_EQ(res, Kokkos::pow(a, a));
      },
      errors);
  EXPECT_EQ(0, errors);

  Kokkos::parallel_reduce(
      "Test_Pow2", Kokkos::RangePolicy(0, ntests),
      KOKKOS_LAMBDA(int i, int &error) {
        Real a = (Real)i;
        Real b = (Real)2;
        Real res = call_variant(fvariant, a, b);
        DEXPECT_EQ(res, Kokkos::pow(a, b));
      },
      errors);
  EXPECT_EQ(0, errors);
}

TEST(Function_Ptr, float) { test_helper_Function_Ptr<float>(); }

TEST(Function_Ptr, double) { test_helper_Function_Ptr<double>(); }

TEST_MAIN
