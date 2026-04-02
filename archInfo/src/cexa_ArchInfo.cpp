// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

#include "cexa_ArchInfo.hpp"

#include <Kokkos_Core.hpp>

#include <string>
#include <ostream>
#include <iostream>

#if defined(UNIX) || defined(__unix__)
#include <cexa_unixArchInfo.hpp>
#elif defined(_WIN32)
#include <cexa_windowsArchInfo.hpp>
#elif defined(__APPLE__)
#include <cexa_macosArchInfo.hpp>
#else
#error "This utility only supports unix, windows and macos"
#endif

namespace cexa {

namespace impl {

// Kokkos can use a subset of the available threads
std::size_t get_kokkos_concurrency() { return Kokkos::num_threads(); }

#if defined(KOKKOS_ENABLE_HIP)

std::string get_gpu_name() {
  hipDeviceProp_t device_prop;
  int device_id = Kokkos::device_id();
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceProperties(&device_prop, device_id));
  return std::string{device_prop.name};
}

std::string get_gpu_arch() {
  hipDeviceProp_t device_prop;
  int device_id = Kokkos::device_id();
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceProperties(&device_prop, device_id));
  return std::string{device_prop.gcnArchName};
}

std::string get_gpu_driver_version() {
  int driver_version = 0;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDriverGetVersion(&driver_version));
  int version_major = driver_version / 10000000;
  int version_minor = (driver_version - version_major * 10000000) / 100000;
  int version_patch =
      driver_version - version_major * 10000000 - version_minor * 100000;

  std::stringstream ss_driver_version;
  ss_driver_version << version_major << "." << version_minor << "."
                    << version_patch;
  return ss_driver_version.str();
}

std::string get_gpu_runtime_version() {
  int runtime_version = 0;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipRuntimeGetVersion(&runtime_version));
  int version_major = runtime_version / 10000000;
  int version_minor = (runtime_version - version_major * 10000000) / 100000;
  int version_patch =
      runtime_version - version_major * 10000000 - version_minor * 100000;

  std::stringstream ss_runtime_version;
  ss_runtime_version << version_major << "." << version_minor << "."
                     << version_patch;
  return ss_runtime_version.str();
}

#elif defined(KOKKOS_ENABLE_CUDA)

#include <nvml.h>

std::string get_gpu_name() {
  cudaDeviceProp device_prop;
  int device_id = Kokkos::device_id();
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, device_id));
  return std::string{device_prop.name};
}

std::string get_gpu_arch() {
  cudaDeviceProp device_prop;
  int device_id = Kokkos::device_id();
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, device_id));

  std::stringstream ss;
  ss << device_prop.major << device_prop.minor;
  return ss.str();
}

std::string get_gpu_driver_version() {
  if (NVML_SUCCESS != nvmlInit_v2()) {
    return "ERROR";
  }
  char buffer[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE];
  nvmlSystemGetDriverVersion(buffer, NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE);
  nvmlShutdown();
  return buffer;
}

std::string get_gpu_runtime_version() {
  int runtime_version = 0;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaRuntimeGetVersion(&runtime_version));
  int version_major = runtime_version / 1000;
  int version_minor = (runtime_version % 1000) / 10;
  std::stringstream ss;
  ss << version_major << "." << version_minor;
  return ss.str();
}

#elif defined(KOKKOS_ENABLE_SYCL)

template <class Info>
std::string get_sycl_info() {
  sycl::device d;
  try {
    d = sycl::device(sycl::gpu_selector_v);
  } catch (sycl::exception const& e) {
    d = sycl::device(sycl::cpu_selector_v);
  }
  return d.get_info<Info>();
}

std::string get_gpu_name() { return get_sycl_info<sycl::info::device::name>(); }

// The GPU architecture isn't currently exposed by the SYCL api (and wouldn't
// make sense for Intel GPUs).
std::string get_gpu_arch() { return "N/A"; }

std::string get_gpu_driver_version() {
  return get_sycl_info<sycl::info::device::driver_version>();
}

std::string get_gpu_runtime_version() {
  return get_sycl_info<sycl::info::device::version>();
}

#elif defined(KOKKOS_ENABLE_OPENACC)

std::string get_gpu_name() {
  const char* name =
      acc_get_property_string(0, acc_device_current, acc_property_name);

  if (!name) {
    return "ERROR";
  }

  return name;
}

// We cannot query this information from the OpenACC api
std::string get_gpu_arch() { return "N/A"; }

std::string get_gpu_driver_version() {
  const char* driver_version =
      acc_get_property_string(0, acc_device_current, acc_property_driver);

  if (!driver_version) {
    return "ERROR";
  }

  return driver_version;
}

// We cannot query this information from the OpenACC api
std::string get_gpu_runtime_version() { return "N/A"; }

#else

std::string get_gpu_name() {
  return "Not compiled with one of the supported GPU backend: CUDA, HIP, SYCL, or OpenACC";
}

std::string get_gpu_arch() { return "N/A"; }

std::string get_gpu_driver_version() { return "N/A"; }

std::string get_gpu_runtime_version() { return "N/A"; }

#endif

}  // namespace impl

void print_host_info(std::ostream& ostream) {
  using namespace cexa::impl;
  ostream << "HOST:\n"
          << "- CPU Model: " << get_cpu_model_name() << '\n'
          << "- Cores per socket: " << get_core_count_per_socket() << '\n'
          << "- Threads per socket: " << get_thread_count_per_socket() << '\n'
          << "- Sockets: " << get_physical_socket_count() << '\n'
          << "- Kokkos Concurrency: " << get_kokkos_concurrency() << std::endl;
}

void print_os_info(std::ostream& ostream) {
  using namespace cexa::impl;
  ostream << "OS:\n"
          << "- Type: " << get_sys_type() << '\n'
          << "- Name: " << get_sys_name() << '\n'
          << "- Kernel: " << get_kernel_version() << std::endl;
}

void print_device_info(std::ostream& ostream) {
  using namespace cexa::impl;
  ostream << "DEVICE:\n"
          << "- Model: " << get_gpu_name() << '\n'
          << "- Arch: " << get_gpu_arch() << '\n'
          << "- Runtime Version: " << get_gpu_runtime_version() << '\n'
          << "- Driver Version: " << get_gpu_driver_version() << std::endl;
}

}  // namespace cexa
