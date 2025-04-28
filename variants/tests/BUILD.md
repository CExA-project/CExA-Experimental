<!--
SPDX-FileCopyrightText: 2025 CExA-project

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

Instruction to build tests:
- Create cmake build dir:
`cmake -B build_dir_name -DKokkos_DIR=${Kokkos_install_dir}`

- Compile
`cmake --build build_dir_name` [-j X]
