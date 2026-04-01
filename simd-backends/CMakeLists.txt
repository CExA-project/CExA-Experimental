# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

cmake_minimum_required(VERSION 3.23)

project(simd-backends VERSION 0.1.0)

find_package(Kokkos 4.5 REQUIRED)

option(
  CEXA_SIMD_ENABLE_SLEEF
  "Use the sleef library to implement the simd math functions"
  OFF
)
option(CEXA_SIMD_ENABLE_SVML "Use the svml to implement the simd math functions" OFF)
option(CEXA_SIMD_ENABLE_TESTS "Enable accuracy tests" OFF)

if(CEXA_SIMD_ENABLE_SLEEF AND CEXA_SIMD_ENABLE_SVML)
  message(FATAL_ERROR "Only one backend can be enabled at a time")
endif()

if(CEXA_SIMD_ENABLE_SLEEF)
  find_package(sleef 3.6 QUIET)
  if(NOT sleef_FOUND)
    include(FetchContent)

    set(SLEEF_BUILD_GNUABI_LIBS OFF)
    set(SLEEF_BUILD_TESTS OFF)
    set(SLEEF_ENABLE_TLFLOAT OFF)
    set(SLEEF_ENABLE_FFTW OFF)
    set(SLEEF_ENABLE_MPFR OFF)
    list(APPEND CMAKE_MESSAGE_INDENT "[sleef] ")
    FetchContent_Declare(
      sleef
      URL https://github.com/shibatch/sleef/archive/refs/tags/3.9.0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      DOWNLOAD_NO_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(sleef)
    list(POP_BACK CMAKE_MESSAGE_INDENT)
  endif()
endif()

if(CEXA_SIMD_ENABLE_SVML AND NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL IntelLLVM)
  message(
    FATAL_ERROR
    "The SVML simd backend is only available with Intel compilers"
  )
endif()

add_subdirectory(src)

if(CEXA_SIMD_ENABLE_TESTS)
  add_subdirectory(test)
endif()

# Install
include(GNUInstallDirs)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  CexaSimdBackendsConfigVersion.cmake
  COMPATIBILITY AnyNewerVersion
)

install(
  TARGETS simd-backends
  EXPORT CexaSimdBackends
  FILE_SET HEADERS
)

install(
  EXPORT CexaSimdBackends
  FILE CexaSimdBackendsTargets.cmake
  NAMESPACE cexa::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaSimdBackends"
)

configure_package_config_file(
  cmake/CexaSimdBackendsConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/CexaSimdBackendsConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaSimdBackends"
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CexaSimdBackendsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CexaSimdBackendsConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaSimdBackends"
)
