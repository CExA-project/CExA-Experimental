cmake_minimum_required(VERSION 3.16)

project(arch-info-install-test)

find_package(Kokkos REQUIRED)
find_package(CexaArchInfo REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE cexa::archInfo Kokkos::kokkos)
