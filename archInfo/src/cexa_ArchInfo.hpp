#ifndef CEXA_EXP_ARCHINFO_HPP
#define CEXA_EXP_ARCHINFO_HPP

#include <ostream>

namespace cexa::experimental {

#if defined(UNIX) || defined(__unix__)

size_t get_physical_socket_count();
size_t get_core_count_per_socket();
size_t get_thread_count_per_socket();

std::string get_cpu_model_name();
std::string get_sysname();
std::string get_sys_type();
std::string get_linux_kernel_version();

#endif  // defined(UNIX) || defined(__unix__)

}  // namespace cexa::experimental

namespace Kokkos {

void print_cpu(std::ostream& ostream);
void print_os(std::ostream& ostream);

}  // namespace Kokkos

#endif  // CEXA_EXP_ARCHINFO_HPP
