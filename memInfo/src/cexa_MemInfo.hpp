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

namespace {
  constexpr int OVERCOMMIT_DISABLED = 2;
  constexpr char COMMITTED_AS_KEY[] = "Committed_AS:";
  constexpr char COMMIT_LIMIT_KEY[] = "CommitLimit:";
  constexpr char MEMINFO_PATH[]     = "/proc/meminfo";
  constexpr char OVERCOMMIT_PATH[]  = "/proc/sys/vm/overcommit_memory";
}

// On some systems, overcommit is disabled, and the kernel does not allow
// memory allocation beyond the commit limit. This means that allocations
// that touch only a small amount of memory are still counted at their full size.
// man proc_sys_vm
bool is_overcommit_limit_set() {
  std::ifstream overcommit_file(OVERCOMMIT_PATH);
  int overcommit_value = 0;

  if (overcommit_file.is_open()) {
    overcommit_file >> overcommit_value;
  } else {
    return false;
  }
  return (overcommit_value == OVERCOMMIT_DISABLED);
}

size_t get_meminfo_value(const char* key) {
  std::ifstream meminfo(MEMINFO_PATH);
  size_t value = 0;
  std::string line;

  if (meminfo.is_open()) {
    while (std::getline(meminfo, line)) {
      if (line.find(key) != std::string::npos) {
        std::istringstream iss(line);
        iss.ignore(256, ':');
        iss >> value;
        value = (iss.fail()) ? 0 : value * 1024;
        break;
      }
    }
  }
  return value;
}

template <typename Space = Kokkos::DefaultExecutionSpace>
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
  if (overcommit_limit) {
    *total = get_meminfo_value(COMMIT_LIMIT_KEY);
    *free = *total - get_meminfo_value(COMMITTED_AS_KEY);
    return;
  }
  struct sysinfo info;
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
