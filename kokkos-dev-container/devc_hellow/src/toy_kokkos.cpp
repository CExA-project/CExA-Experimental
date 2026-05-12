#include <iostream>

#include <Kokkos_Core.hpp>

int main() {
    Kokkos::initialize();
    {
        Kokkos::parallel_for(10, KOKKOS_LAMBDA (int i) {
          std::cout << "Hello world: " << i << std::endl;
        });
    }
    Kokkos::finalize();

    return 0;
}
