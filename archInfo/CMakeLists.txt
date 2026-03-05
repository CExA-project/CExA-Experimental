cmake_minimum_required(VERSION 3.21.1)

project(cexa-experimental-archInfo LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")
find_package(Kokkos REQUIRED)

include(CTest)

add_subdirectory(src)
add_subdirectory(unit_test)

