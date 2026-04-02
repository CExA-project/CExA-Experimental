<!--
SPDX-FileCopyrightText: 2026 CExA-project

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# SIMD backends

This header only library provides an interface to replace the Kokkos math functions for SIMD types with calls to different external libraries.

## Build

The available cmake configuration options are:
- `CEXA_SIMD_ENABLE_SLEEF`: use the sleef library
- `CEXA_SIMD_ENABLE_SVML`: use the Intel SVML library (only available with intel compilers)
- `CEXA_SIMD_ENABLE_TESTS`: build the tests

The sleef backend requires sleef v3.6.0 or later.

## Usage

### CMake

With CMake, you can use the `find_package` command as shown in the
example below. If not installed to a system location, you should provide the
path to the library using the `-DCexaSimdBackends_ROOT=<path to the install location>`
option.

```cmake
cmake_minimum_required(VERSION 3.23)

project(test)

find_package(Kokkos REQUIRED)
find_package(CexaSimdBackends REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Kokkos::kokkos cexa::simd-backends)
```

You can then include `CEXA_SIMD_Backends.hpp` in your code and use the Kokkos simd math
functions as usual.

```cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <CEXA_SIMD_Backends.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::Experimental::simd<float> vec(1.f);
  vec = Kokkos::exp(vec);
}
```

### Manual

As the library is header only, you can direclty include the header
corresponding to your SIMD library of choice directly in your code. In that
case, you should also handle the necessary include and link flags for the
library you choose.

For example, with sleef:
```cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "CEXA_SIMD_SLEEF.hpp"

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::Experimental::simd<float> vec(1.f);
  vec = Kokkos::exp(vec);
}
```
```
g++ main.cpp -I<path/to/sleef/include/dir> -L<path/to/sleef/lib/dir> -lsleef
```
