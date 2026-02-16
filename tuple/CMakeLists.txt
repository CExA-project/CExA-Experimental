# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
cmake_minimum_required(VERSION 3.22)

project(tuple LANGUAGES CXX)

option(CEXA_TUPLE_ENABLE_TESTS "Build the tuple tests" OFF)

find_package(Kokkos REQUIRED)

add_library(cexa.tuple INTERFACE)
# FIXME: FILE_SET is only available with cmake 3.23+
# target_sources(
#   cexa.tuple
#   INTERFACE
#     FILE_SET HEADERS
#     BASE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include"
#     FILES
#       "${CMAKE_CURRENT_LIST_DIR}/include/tuple.hpp"
#       "${CMAKE_CURRENT_LIST_DIR}/include/impl/apply.hpp"
#       "${CMAKE_CURRENT_LIST_DIR}/include/impl/creation.hpp"
#       "${CMAKE_CURRENT_LIST_DIR}/include/impl/helper.hpp"
#       "${CMAKE_CURRENT_LIST_DIR}/include/impl/ignore.hpp"
#       "${CMAKE_CURRENT_LIST_DIR}/include/impl/macros.hpp"
#       "${CMAKE_CURRENT_LIST_DIR}/include/impl/traits.hpp"
#       "${CMAKE_CURRENT_LIST_DIR}/include/impl/tuple.hpp"
#       "${CMAKE_CURRENT_LIST_DIR}/include/impl/tuple_fwd.hpp"
# )
target_include_directories(cexa.tuple INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(cexa.tuple INTERFACE Kokkos::kokkos)
add_library(cexa::tuple ALIAS cexa.tuple)

if(CEXA_TUPLE_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
