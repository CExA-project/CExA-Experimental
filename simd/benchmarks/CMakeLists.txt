# SPDX-FileCopyrightText: 2025 CExA-project
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

CPMAddPackage(
  benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.9.1
  OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
    "BENCHMARK_ENABLE_INSTALL OFF"
    "BENCHMARK_INSTALL_DOCS OFF"
)

check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
check_cxx_compiler_flag("-pthread" COMPILER_SUPPORTS_PTHREAD)

add_library(benchmark_common INTERFACE)
target_link_libraries(
  benchmark_common
  INTERFACE Kokkos::kokkos benchmark::benchmark math-simd
)

if("${COMPILER_SUPPORTS_PTHREAD}")
  target_link_libraries(benchmark_common INTERFACE pthread)
  target_compile_options(benchmark_common INTERFACE -pthread)
endif()

if(COMPILER_SUPPORTS_MARCH_NATIVE)
  target_compile_options(benchmark_common INTERFACE -march=native)
endif()

if("${SIMD_USE_INTEL_FP_PRECISE}" AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL IntelLLVM)
  target_compile_options(benchmark_common INTERFACE -fp-model=precise -O3)
endif()

add_executable(bench_kokkos SIMD_bench_kokkos.cpp)
target_link_libraries(bench_kokkos PRIVATE benchmark_common)

add_executable(bench_kokkos_parallel SIMD_bench_kokkos_parallel.cpp)
target_link_libraries(bench_kokkos_parallel PRIVATE benchmark_common)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL IntelLLVM)
  add_executable(bench_intrinsics SIMD_bench_intrinsics.cpp)
  target_link_libraries(bench_intrinsics PRIVATE benchmark_common)
endif()

if("${SIMD_BUILD_ACCURACY_BENCHMARK}")
  find_package(mpfr)
  if(NOT "${mpfr_FOUND}")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(mpfr REQUIRED IMPORTED_TARGET mpfr)
  endif()

  CPMAddPackage(
    mpreal
    GITHUB_REPOSITORY advanpix/mpreal
    GIT_TAG master
    DOWNLOAD_ONLY YES
  )

  add_executable(bench_accuracy SIMD_bench_accuracy.cpp)
  if(TARGET PkgConfig::mpfr)
    target_link_libraries(bench_accuracy PRIVATE benchmark_common PkgConfig::mpfr)
  else()
    target_include_directories(bench_accuracy PRIVATE mpfr_INCLUDEDIR)
    target_link_libraries(bench_accuracy PRIVATE benchmark_common mpfr_LIBRARIES)
  endif()
  target_include_directories(bench_accuracy PRIVATE "${mpreal_SOURCE_DIR}")
endif()
