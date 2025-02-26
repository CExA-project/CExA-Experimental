#include <cstddef>
#include <Kokkos_Core.hpp>
#include <sstream>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

namespace Kokkos {
namespace Experimental {

// On some systems, overcommit is disabled, and the kernel will not allow memory
// allocation beyond the commit limit. This means that allocations that only touch
// a small amount of memory will be accounted for at their full allocation size.
bool is_overcommit_limit_set() {
  std::ifstream overcommit_file("/proc/sys/vm/overcommit_memory");
  int overcommit_value = 0;

  if (overcommit_file.is_open()) {
    overcommit_file >> overcommit_value;
    overcommit_file.close();
  } else {
    return false;
  }
  return (overcommit_value == 2);
}

size_t get_committed_as() {
  std::ifstream meminfo("/proc/meminfo");
  size_t committed_as = 0;
  char line[256];

  if (meminfo.is_open()) {
    while (meminfo.getline(line, 256)) {
      if (strncmp(line, "Committed_AS:", 13) == 0) {
        std::istringstream iss(line);
        iss.ignore(256, ':');
        iss >> committed_as;
        committed_as = (iss.fail()) ? 0 : committed_as * 1024;
        meminfo.close();
        break;
      }
    }
    meminfo.close();
  }
  return committed_as;
}

size_t get_commitLimit() {
  std::ifstream meminfo("/proc/meminfo");
  size_t commitLimit = 0;
  char line[256];

  if (meminfo.is_open()) {
    while (meminfo.getline(line, 256)) {
      if (strncmp(line, "CommitLimit:", 12) == 0) {
        std::istringstream iss(line);
        iss.ignore(256, ':');
        iss >> commitLimit;
        commitLimit = (iss.fail()) ? 0 : commitLimit * 1024;
        meminfo.close();
        break;
      }
    }
    meminfo.close();
  }
  return commitLimit;
}

template <typename Space = Kokkos::DefaultExecutionSpace::memory_space>
void MemGetInfo(size_t* free, size_t* total) {
  using MemorySpace = typename Space::memory_space;
  MemGetInfo<MemorySpace>(free, total);
}

// Single node memory info
template <>
void MemGetInfo<Kokkos::HostSpace>(size_t* free, size_t* total) {
#ifdef _WIN32
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex) != 0) {
    *free  = statex.ullAvailPhys;
    *total = statex.ullTotalPhys;
  }
#else
  static bool overcommit_limit = is_overcommit_limit_set();
  struct sysinfo info;
  if (overcommit_limit) {
    *total = get_commitLimit();
    *free = *total - get_committed_as();
    return;
  }
  if (sysinfo(&info) == 0) {
    *free  = info.freeram * info.mem_unit;
    *total = info.totalram * info.mem_unit;
    return;
  }
#endif
}

#if defined(KOKKOS_ENABLE_CUDA)
template <>
void MemGetInfo<Kokkos::CudaSpace>(size_t* free, size_t* total) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemGetInfo(free, total));
}
template <>
void MemGetInfo<Kokkos::CudaUVMSpace>(size_t* free, size_t* total) {
  MemGetInfo<Kokkos::HostSpace>(free, total);
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
void MemGetInfo<Kokkos::HIPSpace>(size_t* free, size_t* total) {
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemGetInfo(free, total));
}
template <>
void MemGetInfo<Kokkos::HIPManagedSpace>(size_t* free, size_t* total) {
  MemGetInfo<Kokkos::HostSpace>(free, total);
}
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
void MemGetInfo<Kokkos::SYCLDeviceUSMSpace>(size_t* free, size_t* total) {
  std::vector<sycl::device> devices = Kokkos::Impl::get_sycl_devices();
  for (auto& dev : devices) {
    if (dev.is_gpu()) {
      *total += dev.get_info<sycl::info::device::global_mem_size>();
      // https://github.com/triSYCL/sycl/blob/sycl/unified/master/sycl/doc/extensions/supported/sycl_ext_intel_device_info.md#free-global-memory
      if (dev.has(sycl::aspect::ext_intel_free_memory)) {
        *free += dev.get_info<sycl::ext::intel::info::device::free_memory>();
      }
    }
  }
}
#endif

}  // namespace Experimental
}  // namespace Kokkos
