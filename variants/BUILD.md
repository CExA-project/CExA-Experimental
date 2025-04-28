<!--
SPDX-FileCopyrightText: 2025 CExA-project

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

Instruction to build tests:
- Create cmake build dir:
`cmake -B build_dir_name -DKokkos_DIR=${Kokkos_install_dir} -DCEXA_BUILD_TESTS=ON -DCEXA_BUILD_MPARK_TESTS=ON`

- Compile
`cmake --build build_dir_name` [-j X]

To build examples:
- Create cmake build dir:
`cmake -B build_dir_name -DKokkos_DIR=${Kokkos_install_dir} -DCEXA_BUILD_EXAMPLES=ON`

- Compile
`cmake --build build_dir_name` [-j X]
