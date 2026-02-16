// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  const Kokkos::ScopeGuard kokkos_scope(argc, argv);
  return RUN_ALL_TESTS();
}
