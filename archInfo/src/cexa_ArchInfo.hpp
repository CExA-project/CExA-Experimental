#ifndef CEXA_EXP_ARCHINFO_HPP
#define CEXA_EXP_ARCHINFO_HPP

#include <string>
#include <ostream>
#include <iostream>

namespace cexa::experimental {

size_t get_kokkos_concurrency();

// CPU
std::string get_cpu_model_name();
size_t get_physical_socket_count();
size_t get_core_count_per_socket();
size_t get_thread_count_per_socket();

// OS
std::string get_sys_name();
std::string get_sys_type();
std::string get_kernel_version();

// GPU
std::string get_gpu_name();
std::string get_gpu_arch();
std::string get_gpu_driver_version();
std::string get_gpu_runtime_version();

}  // namespace cexa::experimental

namespace Kokkos {

void print_os(std::ostream& ostream = std::cout);
void print_cpu(std::ostream& ostream = std::cout);
void print_gpu(std::ostream& ostream = std::cout);

}  // namespace Kokkos

#endif  // CEXA_EXP_ARCHINFO_HPP
