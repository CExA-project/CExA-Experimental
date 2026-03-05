#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cexa_ArchInfo.hpp>

namespace CE = cexa::experimental;

#if defined(UNIX) || defined(__unix__)

TEST(ArchInfo, KernelVersion) {
  ASSERT_GT(CE::get_kernel_version().size(), 0);
}

TEST(ArchInfo, SysName) {
  ASSERT_GT(CE::get_sys_name().size(), 0);
}

TEST(ArchInfo, SysType) {
  ASSERT_GT(CE::get_sys_type().size(), 0);
}

TEST(ArchInfo, CPUModelName) {
  ASSERT_GT(CE::get_cpu_model_name().size(), 0);
}

TEST(ArchInfo, SocketCount) {
  ASSERT_GT(CE::get_physical_socket_count(), 0);
}

TEST(ArchInfo, CoreCountPerSocket) {
  ASSERT_GT(CE::get_core_count_per_socket(), 0);
}

TEST(ArchInfo, ThreadCountPerSocket) {
  ASSERT_GT(CE::get_thread_count_per_socket(), 0);
}

#endif  // defined(UNIX) || defined(__unix__)

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();

  Kokkos::print_cpu();
  Kokkos::print_os();

  Kokkos::finalize();
  return result;
}

