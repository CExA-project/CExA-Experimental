#include "cexa_ArchInfo.hpp"
#include "cexa_compilInfo.hpp"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <ostream>
#include <iostream>

namespace cexa::experimental {

// Kokkos can use a subset of the available threads
size_t get_kokkos_concurrency() { return Kokkos::num_threads(); }

// Non-Linux
#if !(defined(UNIX) || defined(__unix__))

size_t get_physical_socket_count() { return 1; }

size_t get_core_count_per_socket() { return CExA::CompilInfo::CoresCount; }

size_t get_thread_count_per_socket() { return CExA::CompilInfo::ThreadsCount; }

std::string get_cpu_model_name() {
  return std::string{CExA::CompilInfo::Processor};
}

std::string get_sys_name() { return std::string{CExA::CompilInfo::SysName}; }

std::string get_sys_type() { return std::string{CExA::CompilInfo::SysType}; }

std::string get_kernel_version() {
  return std::string{CExA::CompilInfo::SysVersion};
}

// No GPU support
#endif  // !(defined(UNIX) || defined(__unix__))

#if !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_CUDA) && \
    !defined(KOKKOS_ENABLE_SYCL)

std::string get_gpu_name() {
  return std::string{"Not compiled with GPU support"};
}

std::string get_gpu_arch() { return std::string{"N/A"}; }

std::string get_gpu_driver_version() { return std::string{"N/A"}; }

std::string get_gpu_runtime_version() { return std::string{"N/A"}; }

#endif

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
  // Format
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
  // Format
  std::stringstream ss_runtime_version;
  ss_runtime_version << version_major << "." << version_minor << "."
                     << version_patch;
  return ss_runtime_version.str();
}

#elif defined(KOKKOS_ENABLE_CUDA)

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
  int driver_version = 0;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaDriverGetVersion(&driver_version));
  int version_major = driver_version / 1000;
  int version_minor = (driver_version % 1000) / 10;
  std::stringstream ss;
  ss << version_major << "." << version_minor;
  return ss.str();
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

// TODO: SYCL

#endif

}  // namespace cexa::experimental

namespace Kokkos {

void print_cpu(std::ostream& ostream) {
  using namespace cexa::experimental;
  ostream << "CPU Model          : " << get_cpu_model_name() << "\n"
          << "    Cores          : " << get_core_count_per_socket() << "\n"
          << "    Threads        : " << get_thread_count_per_socket() << "\n"
          << "    Sockets        : " << get_physical_socket_count() << "\n"
          << "Kokkos Concurrency : " << get_kokkos_concurrency() << std::endl;
}

void print_os(std::ostream& ostream) {
  using namespace cexa::experimental;
  ostream << "OS Type   : " << get_sys_type() << "\n"
          << "   Name   : " << get_sys_name() << "\n"
          << "   Kernel : " << get_kernel_version() << std::endl;
}

void print_gpu(std::ostream& ostream) {
  using namespace cexa::experimental;
  ostream << "GPU Model           :" << get_gpu_name() << "\n"
          << "    Arch            :" << get_gpu_arch() << "\n"
          << "    Runtime Version :" << get_gpu_runtime_version() << "\n"
          << "    Driver Version  :" << get_gpu_driver_version() << std::endl;
}

}  // namespace Kokkos
