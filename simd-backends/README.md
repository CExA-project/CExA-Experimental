# SIMD backends

This header only library provides an interface to replace the Kokkos math functions for SIMD types with calls to different external libraries.

## Build

The available cmake configuration options are:
- `ENABLE_SLEEF`: use the sleef library
- `ENABLE_SVML`: use the Intel SVML library (only available with intel compilers)
- `ENABLE_TESTS`: build the tests
- `ENABLE_INSTALL`: enable the installation target

## Usage

### Manual

As the library is header only, you can include the header for your library of choice
directly in your code. Or include `Kokkos_SIMD_Backends.hpp` and use the macros
`KOKKOS_ENABLE_<backend_name>` to control which library is being used.

### CMake

With CMake, you can use the `find_package` command as shown in the
example below. If not installed to a system location, you should provide the
path to the library using the `-DKokkosSimdBackends_ROOT=<path to the install location>`
option.

```cmake
cmake_minimum_required(VERSION 3.16)

project(test)

find_package(Kokkos 4.5 REQUIRED CONFIG)
find_package(KokkosSimdBackends 0.1 REQUIRED CONFIG)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE Kokkos::kokkos cexa-experimental::simd-backends)
```

You can then include `Kokkos_SIMD_BACKENDS.hpp` in your code and use the Kokkos simd math
functions as usual.
