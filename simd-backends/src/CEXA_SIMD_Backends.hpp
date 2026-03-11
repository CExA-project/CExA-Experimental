// SPDX-FileCopyrightText: 2025 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#ifndef CEXA_SIMD_BACKENDS_HPP
#define CEXA_SIMD_BACKENDS_HPP

#include <Kokkos_SIMD.hpp>

#if defined(CEXA_ENABLE_SLEEF)
#include <CEXA_SIMD_SLEEF.hpp>
#elif defined(CEXA_ENABLE_SVML)
#include <CEXA_SIMD_SVML.hpp>
#endif

#endif
