cmake_minimum_required(VERSION 3.16)

project(cexa-experimental-simd)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")

find_package(Kokkos REQUIRED)

add_subdirectory(src)
add_subdirectory(unit_tests)
add_subdirectory(benchmarks)
