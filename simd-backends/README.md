# SIMD backends

This header only library provides an interface to replace the Kokkos math functions for SIMD types with calls to different external libraries.

## Build

To use the library, you can use the `find_package` command as shown in the
example below. If not installed to a system location, you should provide the
path to the library using the `-DKokkosSimdBackends_ROOT#<path to the lib>`
option.

```cmake
cmake_minimum_required(VERSION 3.16)

project(test)

find_package(Kokkos 4.5 REQUIRED CONFIG)
find_package(KokkosSimdBackends 0.1 REQUIRED CONFIG)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE Kokkos::kokkos cexa-experimental::simd-backends)
```

## Supported backends

The supported SIMD math libraries are:
- [Sleef](https://sleef.org/)
- Intel SVML (only available with intel compilers)
