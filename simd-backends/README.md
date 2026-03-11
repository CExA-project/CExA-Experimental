# SIMD backends

This header only library provides an interface to replace the Kokkos math functions for SIMD types with calls to different external libraries.

## Build

The available cmake configuration options are:
- `CEXA_SIMD_ENABLE_SLEEF`: use the sleef library
- `CEXA_SIMD_ENABLE_SVML`: use the Intel SVML library (only available with intel compilers)
- `CEXA_SIMD_ENABLE_TESTS`: build the tests
- `CEXA_SIMD_ENABLE_INSTALL`: enable the installation target

## Usage

### Manual

As the library is header only, you can include the header for your library of choice
directly in your code. Or include `CEXA_SIMD_Backends.hpp` and use the macros
`CEXA_SIMD_ENABLE_<backend_name>` to control which library is being used.

### CMake

With CMake, you can use the `find_package` command as shown in the
example below. If not installed to a system location, you should provide the
path to the library using the `-DCexaSimdBackends_ROOT=<path to the install location>`
option.

```cmake
cmake_minimum_required(VERSION 3.16)

project(test)

find_package(Kokkos 4.5 REQUIRED CONFIG)
find_package(CexaSimdBackends 0.1 REQUIRED CONFIG)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Kokkos::kokkos cexa-experimental::simd-backends)
```

You can then include `CEXA_SIMD_Backends.hpp` in your code and use the Kokkos simd math
functions as usual.
