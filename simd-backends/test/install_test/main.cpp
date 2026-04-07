#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <CEXA_SIMD_Backends.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::Experimental::simd<float> vec(1.f);
  vec = Kokkos::exp(vec);

  return 0;
}
