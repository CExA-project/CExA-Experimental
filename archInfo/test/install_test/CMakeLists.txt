# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

cmake_minimum_required(VERSION 3.16)

project(arch-info-install-test)

find_package(Kokkos REQUIRED)
find_package(CexaArchInfo REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE cexa::archInfo Kokkos::kokkos)
