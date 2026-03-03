#ifndef KOKKOS_MEMINFO_HPP
#define KOKKOS_MEMINFO_HPP

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstddef>

#include <Kokkos_Core.hpp>

#ifndef _WIN32
#include <unixMemInfo.hpp>
#endif

namespace Kokkos::Experimental {

template <typename Space = Kokkos::DefaultExecutionSpace>
void MemGetInfo(size_t* free, size_t* total) {
  using MemorySpace = typename Space::memory_space;
  MemGetInfo<MemorySpace>(free, total);
}

// Single node memory info
#ifdef _WIN32
template <>
inline void MemGetInfo<Kokkos::HostSpace>(size_t* free, size_t* total) {
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex) != 0) {
    *free  = statex.ullAvailPhys;
    *total = statex.ullTotalPhys;
  }
}
#endif

#if defined(KOKKOS_ENABLE_CUDA)
template <>
inline void MemGetInfo<Kokkos::CudaSpace>(size_t* free, size_t* total) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemGetInfo(free, total));
}
template <>
inline void MemGetInfo<Kokkos::CudaUVMSpace>(size_t* free, size_t* total) {
  MemGetInfo<Kokkos::HostSpace>(free, total);
}
template <>
inline void MemGetInfo<Kokkos::CudaHostPinnedSpace>(size_t* free,
                                                    size_t* total) {
  MemGetInfo<Kokkos::HostSpace>(free, total);
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
inline void MemGetInfo<Kokkos::HIPSpace>(size_t* free, size_t* total) {
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemGetInfo(free, total));
}
template <>
inline void MemGetInfo<Kokkos::HIPManagedSpace>(size_t* free, size_t* total) {
  MemGetInfo<Kokkos::HostSpace>(free, total);
}
template <>
inline void MemGetInfo<Kokkos::HIPHostPinnedSpace>(size_t* free,
                                                   size_t* total) {
  MemGetInfo<Kokkos::HostSpace>(free, total);
}
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
inline void MemGetInfo<Kokkos::SYCLDeviceUSMSpace>(size_t* free,
                                                   size_t* total) {
  std::vector<sycl::device> devices = Kokkos::Impl::get_sycl_devices();
  if (devices.empty()) {
    return;
  }
  int device_id = Kokkos::Impl::SYCLInternal::m_syclDev;
  if (device_id < 0 || device_id >= static_cast<int>(devices.size())) {
    return;
  }
  auto device = devices[Impl::SYCLInternal::m_syclDev];

  *total = 0;
  *free  = 0;
  if (device.is_gpu()) {
    if (device.has(sycl::aspect::ext_intel_free_memory)) {
      *free  = device.get_info<sycl::ext::intel::info::device::free_memory>();
      *total = device.get_info<sycl::info::device::global_mem_size>();
    }
  }
}
#endif

}  // namespace Kokkos::Experimental

#endif  // KOKKOS_MEMINFO_HPP
