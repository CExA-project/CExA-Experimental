# SPDX-FileCopyrightText: 2025 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

cmake_minimum_required(VERSION 3.21.1)

project(Cexa.Variant LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_library(Cexa.Variant INTERFACE)
target_include_directories(Cexa.Variant INTERFACE include/)

option(CEXA_BUILD_TESTS "Build tests" OFF)
option(CEXA_BUILD_MPARK_TESTS "Build the tests from MPark::variant" OFF)

if("${CEXA_BUILD_MPARK_TESTS}" AND NOT "${CEXA_BUILD_TESTS}")
  message(FATAL_ERROR "Can't build MPark tests if testing is not enabled (CEXA_BUILD_TESTS should be ON)")
endif()

if("${CEXA_BUILD_TESTS}")
  enable_testing()
  add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/googletest)
  add_subdirectory(test/)
endif()


