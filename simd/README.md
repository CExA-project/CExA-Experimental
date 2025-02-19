<!--
SPDX-FileCopyrightText: 2025 CExA-project

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# SIMD

Implementation of the `exp` function for AVX2 and AVX512.

## Usage

# Using the functions directly

The functions can be used direclty by including the files
[AVX2_Math.hpp](./src/AVX2_Math.hpp) and
[AVX512_Math.hpp](./src/AVX512_Math.hpp). The following functions are defined
in the namespace `Cexa::Experimental::simd`:
- `__m128 avx2::exp4f(__m128)`
- `__m256 avx2::exp8f(__m256)`
- `__m256d avx2::exp4d(__m256d)`
- `__m256 avx512::exp8f(__m256)`
- `__m512 avx512::exp16f(__m512)`
- `__m512d avx512::exp8d(__m512d)`

# Using the functions as `Kokkos::exp`

These functions can be made available as `Kokkos::exp` overloads by including
the file [Kokkos_SIMD_AVX_Math.hpp](./src/Kokkos_SIMD_AVX_Math.hpp).

```c++
#include <Kokkos_SIMD_AVX_Math.hpp>
```

Note that when compiling with the Intel C++ compiler, intel's intrinsics for the
exponential will be used instead of these functions.

## Building

The core functions are only defined in the header files in the [src](./src)
directory, so these can be copied directly in your project. Additionnaly, tests
and benchmarks can be built using cmake, the relevant options are:
- `SIMD_BUILD_TESTS`: Build the tests (default: `ON`)
- `SIMD_BUILD_BENCHMARKS`: Build the benchmarks (default: `ON`)
- `SIMD_USE_INTEL_FP_PRECISE`: Use the flag `-fp-model=precise` when compiling
  with an Intel compiler, its default value is `fast=1` which enables
  floating point math optimizations which might hurt accuracy (default: `OFF`) 
- `SIMD_BUILD_ACCURACY_BENCHMARK`: Build the accuracy benchmark, requires mpfr
  to be installed (default: `ON`)
