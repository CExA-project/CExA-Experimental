cmake_minimum_required(VERSION 3.21)

project(KokkosWithOpenMP)

find_package(Kokkos 5.1 REQUIRED CONFIG)

add_executable(toy_kokkos src/toy_kokkos.cpp)
target_link_libraries(toy_kokkos PRIVATE Kokkos::kokkos)
