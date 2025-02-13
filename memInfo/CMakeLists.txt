cmake_minimum_required(VERSION 3.16)

project(cexa-experimental-meminfo LANGUAGES CXX)
set(CMAKE_BUILD_TYPE "RelwithDebInfo")

if (NOT ${Kokkos_DIR} STREQUAL "")
    add_subdirectory(${Kokkos_DIR})
    include_directories(${Kokkos_INCLUDE_DIRS_RET})
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")

include(CTest)

find_package(Kokkos REQUIRED)

add_subdirectory(src)
add_subdirectory(unit_test)

