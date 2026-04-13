cmake_minimum_required(VERSION 3.21)

project(KokkosWithOpenMP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

find_package(Kokkos 4.6 REQUIRED CONFIG)

add_executable(toy_kokkos src/toy_kokkos.cpp)
target_link_libraries(toy_kokkos PRIVATE Kokkos::kokkos)
