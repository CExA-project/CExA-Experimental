# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

cmake_minimum_required(VERSION 3.16)

project(simd-backends VERSION 0.1.0)

find_package(Kokkos 4.5 REQUIRED CONFIG)

option(
  CEXA_SIMD_ENABLE_SLEEF
  "Use the sleef library to implement the simd math functions"
  OFF
)
option(CEXA_SIMD_ENABLE_SVML "Use the svml to implement the simd math functions" OFF)
option(CEXA_SIMD_ENABLE_TESTS "Enable accuracy tests" OFF)
option(CEXA_SIMD_ENABLE_INSTALL "Enable installation of the library" ON)

if(CEXA_SIMD_ENABLE_SLEEF AND CEXA_SIMD_ENABLE_SVML)
  message(FATAL_ERROR "Only one backend can be enabled at a time")
endif()

include(FetchContent)
include(GNUInstallDirs)

set(USING_USER_SLEEF FALSE)
if(CEXA_SIMD_ENABLE_SLEEF)
  find_package(sleef 3.6 QUIET CONFIG)
  if(sleef_FOUND)
    set(USING_USER_SLEEF TRUE)
  else()
    set(SLEEF_BUILD_GNUABI_LIBS OFF)
    set(SLEEF_BUILD_TESTS OFF)
    set(SLEEF_ENABLE_TLFLOAT OFF)
    set(SLEEF_ENABLE_FFTW OFF)
    set(SLEEF_ENABLE_MPFR OFF)
    FetchContent_Declare(
      sleef
      URL https://github.com/shibatch/sleef/archive/refs/tags/3.9.0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      DOWNLOAD_NO_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(sleef)
  endif()
endif()

if(CEXA_SIMD_ENABLE_SVML AND NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL IntelLLVM)
  message(
    FATAL_ERROR
    "The SVML simd backend is only available with Intel compilers"
  )
endif()

add_library(simd-backends INTERFACE)
target_include_directories(
  simd-backends
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(simd-backends INTERFACE Kokkos::kokkossimd)

set(
  SIMD_BACKENDS_HEADERS
  src/CEXA_SIMD_Backends.hpp
  src/CEXA_SIMD_SLEEF.hpp
  src/CEXA_SIMD_SVML.hpp
)

if(CEXA_SIMD_ENABLE_SLEEF)
  target_link_libraries(simd-backends INTERFACE sleef)
  target_compile_definitions(simd-backends INTERFACE CEXA_ENABLE_SLEEF)
elseif(CEXA_SIMD_ENABLE_SVML)
  target_compile_definitions(simd-backends INTERFACE CEXA_ENABLE_SVML)
endif()

if(CEXA_SIMD_ENABLE_TESTS)
  add_subdirectory(test)
endif()

if(CEXA_SIMD_ENABLE_INSTALL)
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    CexaSimdBackendsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  add_library(cexa-experimental::simd-backends ALIAS simd-backends)

  install(
    TARGETS simd-backends
    EXPORT CexaSimdBackends
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )

  install(
    FILES ${SIMD_BACKENDS_HEADERS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )

  install(
    EXPORT CexaSimdBackends
    FILE CexaSimdBackendsTargets.cmake
    NAMESPACE cexa-experimental::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaSimdBackends"
  )

  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/CexaSimdBackendsConfig.cmake.in"
    CexaSimdBackendsConfig.cmake
    @ONLY
  )
  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/CexaSimdBackendsConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/CexaSimdBackendsConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaSimdBackends"
  )
endif()
