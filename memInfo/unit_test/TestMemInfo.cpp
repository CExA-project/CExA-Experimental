#include <iostream>
#include <cstddef>
#include <Kokkos_Core.hpp>
#include <cexa_MemInfo.hpp>
#include <gtest/gtest.h>

template <typename MemorySpace>
void testMemInfo() {
  size_t step1_dram = 0ul;
  size_t step2_dram = 0ul;
  size_t step2_freeDram = 0ul;
  size_t step1_freeDram = 0ul;
  Kokkos::Experimental::MemGetInfo<MemorySpace>(&step1_freeDram, &step1_dram);
  {
    Kokkos::View<double*, MemorySpace> data("1 GiB", 1024*1024*1024);
    Kokkos::Experimental::MemGetInfo<MemorySpace>(&step2_freeDram, &step2_dram);
  }
  // Same total memory before and after allocating 1 GiB
  EXPECT_EQ(step1_dram, step2_dram);
  // Check that free memory is less after allocating 1 GiB 
  EXPECT_LT(step2_freeDram, step1_freeDram);
}

TEST(TestMemInfo, testHost) {
  testMemInfo<Kokkos::HostSpace>();
}

#if defined(KOKKOS_ENABLE_CUDA)
TEST(TestMemInfo, testCuda) {
  testMemInfo<Kokkos::CudaSpace>();
}
TEST(TestMemInfo, testCudaUVM) {
  testMemInfo<Kokkos::SharedSpace>();
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
TEST(TestMemInfo, testHip) {
  testMemInfo<Kokkos::HIPSpace>();
}
TEST(TestMemInfo, testHipManaged) {
  testMemInfo<Kokkos::SharedSpace>();
}
#endif

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}