#include <cstddef>
#include <Kokkos_Core.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

namespace Kokkos {
namespace Experimental {

template <typename Space>
void MemGetInfo(size_t* free, size_t* total) {
  *free  = 0;
  *total = 0;
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
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    *free  = info.freeram * info.mem_unit;
    *total = info.totalram * info.mem_unit;
  }
#endif
}

#if defined(KOKKOS_ENABLE_CUDA)
template <>
void MemGetInfo<Kokkos::CudaSpace>(size_t* free, size_t* total) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemGetInfo(free, total));
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
void MemGetInfo<Kokkos::HIPSpace>(size_t* free, size_t* total) {
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemGetInfo(free, total));
}
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
void MemGetInfo<Kokkos::SYCLDeviceUSMSpace>(size_t* free, size_t* total) {
  std::vector<sycl::device> devices = Kokkos::Impl : get_sycl_devices();
  for (auto& dev : devices) {
    if (dev.is_gpu()) {
      *total += dev.get_info<sycl::info::device::global_mem_size>();
      // https://github.com/triSYCL/sycl/blob/sycl/unified/master/sycl/doc/extensions/supported/sycl_ext_intel_device_info.md#free-global-memory
      if (dev.has(aspect::ext_intel_free_memory)) {
        *free += dev.get_info<ext::intel::info::device::free_memory>();
      }
    }
  }
}
#endif

}  // namespace Experimental
}  // namespace Kokkos
