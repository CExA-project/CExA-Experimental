// SPDX-FileCopyrightText: 2025 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#ifndef KOKKOS_SIMD_BACKENDS_HPP
#define KOKKOS_SIMD_BACKENDS_HPP

#include <Kokkos_SIMD.hpp>

#if defined(KOKKOS_ENABLE_SLEEF)
#include <Kokkos_SIMD_SLEEF.hpp>
#elif defined(KOKKOS_ENABLE_SVML)
#include <Kokkos_SIMD_SVML.hpp>
#endif

#endif
