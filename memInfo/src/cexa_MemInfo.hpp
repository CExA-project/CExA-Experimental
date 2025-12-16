#ifndef KOKKOS_MEMINFO_HPP
#define KOKKOS_MEMINFO_HPP

#ifdef _WIN32
#include <windows.h>
#else
#include <unixMemInfo.hpp>
#endif

#include <cstddef>
#include <sstream>
#include <fstream>

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {

template <typename Space = Kokkos::DefaultExecutionSpace>
void MemGetInfo(size_t* free, size_t* total) {
  using MemorySpace = typename Space::memory_space;
  MemGetInfo<MemorySpace>(free, total);
}

// Single node memory info
#ifdef _WIN32
template <>
void MemGetInfo<Kokkos::HostSpace>(size_t* free, size_t* total) {
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex) != 0) {
    *free  = statex.ullAvailPhys;
    *total = statex.ullTotalPhys;
  }
  return;
}
#endif

#if defined(KOKKOS_ENABLE_CUDA)
template <>
void MemGetInfo<Kokkos::CudaSpace>(size_t* free, size_t* total) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemGetInfo(free, total));
}
template <>
void MemGetInfo<Kokkos::CudaUVMSpace>(size_t* free, size_t* total) {
  MemGetInfo<Kokkos::HostSpace>(free, total);
}
template <>
void MemGetInfo<Kokkos::CudaHostPinnedSpace>(size_t* free, size_t* total) {
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
template <>
void MemGetInfo<Kokkos::HIPHostPinnedSpace>(size_t* free, size_t* total) {
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

#endif  // KOKKOS_MEMINFO_HPP
