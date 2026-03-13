#ifndef CEXA_ARCHINFO_HPP
#define CEXA_ARCHINFO_HPP

#include <string>
#include <ostream>
#include <iostream>

namespace cexa {

namespace impl {

std::size_t get_kokkos_concurrency();

// CPU
std::string get_cpu_model_name();
std::size_t get_physical_socket_count();
std::size_t get_core_count_per_socket();
std::size_t get_thread_count_per_socket();

// OS
std::string get_sys_name();
std::string get_sys_type();
std::string get_kernel_version();

// GPU
std::string get_gpu_name();
std::string get_gpu_arch();
std::string get_gpu_driver_version();
std::string get_gpu_runtime_version();

}  // namespace impl

void print_os_info(std::ostream& ostream = std::cout);
void print_host_info(std::ostream& ostream = std::cout);
void print_device_info(std::ostream& ostream = std::cout);

}  // namespace cexa

#endif  // CEXA_EXP_ARCHINFO_HPP
