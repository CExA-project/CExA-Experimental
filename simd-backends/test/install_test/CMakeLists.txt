# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

cmake_minimum_required(VERSION 3.16)

project(simd-backends-install-test)

find_package(Kokkos 5 REQUIRED)
find_package(CexaSimdBackends REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE cexa::simd-backends Kokkos::kokkos)
