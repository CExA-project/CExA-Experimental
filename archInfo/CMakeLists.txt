cmake_minimum_required(VERSION 3.21.1)

project(cexa-experimental-archInfo LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")
find_package(Kokkos REQUIRED)

option(CEXA_ARCHINFO_ENABLE_TESTS "Build the unit tests" OFF)

add_subdirectory(src)

if (CEXA_ARCHINFO_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(unit_test)
endif()
