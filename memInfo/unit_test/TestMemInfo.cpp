#include <cexa_MemInfo.hpp>

#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

template <typename Space = Kokkos::DefaultExecutionSpace>
void testMemInfo() {
  using memory_space = typename std::conditional<
    Kokkos::is_memory_space<Space>::value,
    Space,
    typename Space::memory_space
  >::type;

  std::size_t step1_total = 0ull;
  std::size_t step2_total = 0ull;
  std::size_t step1_free = 0ull;
  std::size_t step2_free = 0ull;

  Kokkos::Experimental::MemGetInfo<memory_space>(&step1_free, &step1_total);

  // Allocate 64 MiB of memory
  Kokkos::View<double*, memory_space> data("data test", 1024*8192);
  Kokkos::fence();
  Kokkos::Experimental::MemGetInfo<memory_space>(&step2_free, &step2_total);

  // Same total memory before and after aloccation
  EXPECT_EQ(step1_total, step2_total);
  // Check that free memory is less after allocation
  const bool is_shared_space = std::is_same<memory_space, Kokkos::SharedSpace>::value;
  if (is_shared_space) {
    EXPECT_LE(step2_free, step1_free);
  } else {
    EXPECT_LT(step2_free, step1_free);
  }
}

#define TEST_SPACE(Space)                                           \
  TEST(MemInfo, Space) {                                            \
    testMemInfo<Kokkos::Space>();                                   \
  }

TEST(MemInfo, DefaultSpace) {
  testMemInfo<>();
}

TEST_SPACE(HostSpace)

#if defined(KOKKOS_ENABLE_CUDA)
  TEST_SPACE(CudaSpace)
  TEST_SPACE(SharedSpace)
  TEST_SPACE(SharedHostPinnedSpace)
#endif

#if defined(KOKKOS_ENABLE_HIP)
  TEST_SPACE(HIPSpace)
  TEST_SPACE(SharedSpace)
  TEST_SPACE(SharedHostPinnedSpace)
#endif

#if defined(KOKKOS_ENABLE_SYCL)
  TEST_SPACE(SYCLDeviceUSMSpace)
  TEST_SPACE(SharedSpace)
#endif

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}

#undef TEST_SPACE
