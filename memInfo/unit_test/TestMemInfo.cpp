#include <Kokkos_Core.hpp>
#include <cexa_MemInfo.hpp>
#include <gtest/gtest.h>
#include <cstddef>

template <typename MemorySpace = void>
void testMemInfo() {
  size_t step1_total = 0ul;
  size_t step2_total = 0ul;
  size_t step2_free = 0ul;
  size_t step1_free = 0ul;

  if constexpr (std::is_same<MemorySpace, void>::value) {
    Kokkos::Experimental::MemGetInfo(&step1_free, &step1_total);
  } else {
    Kokkos::Experimental::MemGetInfo<MemorySpace>(&step1_free, &step1_total);
  }
  volatile double k = 0.0;
  {
    // Allocate 128 MiB of memory
    Kokkos::View<double**, MemorySpace> data("data test", 128, 1024*1024);
    if constexpr (std::is_same<MemorySpace, void>::value) {
      Kokkos::Experimental::MemGetInfo(&step2_free, &step2_total);
    } else {
      Kokkos::Experimental::MemGetInfo<MemorySpace>(&step2_free, &step2_total);
    }
  }
  // Same total memory before and after aloccation
  EXPECT_EQ(step1_total, step2_total);
  // Check that free memory is less after allocation
  EXPECT_LT(step2_free, step1_free);
}

TEST(MemInfo, HostSpace) {
  Kokkos::initialize();
  testMemInfo<Kokkos::HostSpace>();
  Kokkos::finalize();
}
TEST(MemInfo, HostSpaceUninitialized) {
  size_t free = 0;
  size_t total = 0;
  Kokkos::Experimental::MemGetInfo<Kokkos::HostSpace>(&free, &total);
  EXPECT_GT(free, 0);
  EXPECT_GT(total, 0);
}

TEST(MemInfo, DefaultSpace) {
  Kokkos::initialize();
  testMemInfo<>();
  Kokkos::finalize();
}
TEST(MemInfo, DefaultSpaceUninitialized) {
  size_t free = 0;
  size_t total = 0;
  Kokkos::Experimental::MemGetInfo(&free, &total);
  EXPECT_GT(free, 0);
  EXPECT_GT(total, 0);
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST(MemInfo, CudaSpace) {
  Kokkos::initialize();
  testMemInfo<Kokkos::CudaSpace>();
  testMemInfo<Kokkos::SharedSpace>();
  Kokkos::finalize();
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
TEST(MemInfo, HIPSpace) {
  Kokkos::initialize();
  testMemInfo<Kokkos::HIPSpace>();
  testMemInfo<Kokkos::SharedSpace>();
  Kokkos::finalize();
}
#endif

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  return result;
}