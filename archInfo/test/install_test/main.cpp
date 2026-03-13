#include <Kokkos_Core.hpp>
#include <cexa_ArchInfo.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  cexa::print_os_info();
  cexa::print_host_info();
  cexa::print_device_info();
}
