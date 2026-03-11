#include <Kokkos_Core.hpp>
#include <cexa_ArchInfo.hpp>

int main(int argc, char* argv[]) {
	Kokkos::ScopeGuard kokkos_scope(argc, argv);

	Kokkos::print_os();
	Kokkos::print_cpu();
	Kokkos::print_gpu();
}
