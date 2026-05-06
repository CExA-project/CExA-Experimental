#pragma once

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#define CEXA_IMPL_STRINGIFY_VALUE(x) KOKKOS_IMPL_STRINGIFY(x)

#define CEXA_EXPECT(v)                                  \
  if (!static_cast<bool>((v)))                          \
  Kokkos::abort(__FILE__ ":" CEXA_IMPL_STRINGIFY_VALUE( \
      __LINE__) ": Assertion failed: " #v " evaluates to false")

#define CEXA_EXPECT_EQ(a, b)                            \
  if (!((a) == (b)))                                    \
  Kokkos::abort(__FILE__ ":" CEXA_IMPL_STRINGIFY_VALUE( \
      __LINE__) ": Assertion failed: " #a " == " #b)

#if defined(CEXA_IMPL_HOST_TEST)
#define CEXA_IMPL_TEST(suite_name, test_name) TEST(host_##suite_name, test_name)
#else
#define CEXA_IMPL_TEST(suite_name, test_name) \
  TEST(device_##suite_name, test_name)
#endif

#define CEXA_TEST(suite_name, test_name, expr)                                \
  void suite_name##_##test_name##_helper() {                                  \
    Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),           \
                         KOKKOS_LAMBDA(int){KOKKOS_IMPL_STRIP_PARENS(expr)}); \
    Kokkos::fence();                                                          \
  }                                                                           \
  CEXA_IMPL_TEST(suite_name, test_name) { suite_name##_##test_name##_helper(); }

#if (defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)) ||         \
    (defined(KOKKOS_ENABLE_HIP) && defined(__HIP_DEVICE_COMPILE__)) || \
    (defined(KOKKOS_ENABLE_SYCL) && defined(__SYCL_DEVICE_ONLY__))
#define CEXA_ON_DEVICE
#endif

#if defined(KOKKOS_ENABLE_CUDA)
#define CEXA_HOST_DEVICE_NVCC_WARNINGS_PUSH() \
  _Pragma("nv_diagnostic push")               \
      _Pragma("nv_diag_suppress 20011,20013,20014,20015")

#define CEXA_HOST_DEVICE_NVCC_WARNINGS_POP() _Pragma("nv_diagnostic pop")
#else
#define CEXA_HOST_DEVICE_NVCC_WARNINGS_PUSH()
#define CEXA_HOST_DEVICE_NVCC_WARNINGS_POP()
#endif
