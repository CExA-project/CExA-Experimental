# SPDX-FileCopyrightText: 2025 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
cmake_minimum_required(VERSION 3.16)

project(simd-backends VERSION 0.1.0)

find_package(Kokkos 4.5 REQUIRED CONFIG)

option(
  ENABLE_SLEEF
  "Use the sleef library to implement the simd math functions"
  OFF
)
option(ENABLE_SVML "Use the svml to implement the simd math functions" OFF)
option(ENABLE_TESTS "Enable accuracy tests" OFF)
option(ENABLE_INSTALL "Enable installation of the library" ON)

if(ENABLE_SLEEF AND ENABLE_SVML)
  message(FATAL_ERROR "Only one backend can be enabled at a time")
endif()

include(FetchContent)
include(GNUInstallDirs)

set(USING_USER_SLEEF FALSE)
if(ENABLE_SLEEF)
  find_package(sleef 3.6 QUIET CONFIG)
  if(sleef_FOUND)
    set(USING_USER_SLEEF TRUE)
  else()
    set(SLEEF_BUILD_GNUABI_LIBS OFF)
    set(SLEEF_BUILD_TESTS OFF)
    FetchContent_Declare(
      sleef
      URL https://github.com/shibatch/sleef/archive/refs/tags/3.9.0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      DOWNLOAD_NO_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(sleef)
  endif()
endif()

if(ENABLE_SVML AND NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL IntelLLVM)
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
  src/Kokkos_SIMD_Backends.hpp
  src/Kokkos_SIMD_SLEEF.hpp
  src/Kokkos_SIMD_SVML.hpp
)

if(ENABLE_SLEEF)
  target_link_libraries(simd-backends INTERFACE sleef)
  target_compile_definitions(simd-backends INTERFACE KOKKOS_ENABLE_SLEEF)
elseif(ENABLE_SVML)
  target_compile_definitions(simd-backends INTERFACE KOKKOS_ENABLE_SVML)
endif()

if(ENABLE_TESTS)
  add_subdirectory(test)
endif()

if(ENABLE_INSTALL)
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    KokkosSimdBackendsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  add_library(cexa-experimental::simd-backends ALIAS simd-backends)

  install(
    TARGETS simd-backends
    EXPORT KokkosSimdBackends
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )

  install(
    FILES ${SIMD_BACKENDS_HEADERS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )

  install(
    EXPORT KokkosSimdBackends
    FILE KokkosSimdBackendsTargets.cmake
    NAMESPACE cexa-experimental::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KokkosSimdBackends"
  )

  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/KokkosSimdBackendsConfig.cmake.in"
    KokkosSimdBackendsConfig.cmake
    @ONLY
  )
  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/KokkosSimdBackendsConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/KokkosSimdBackendsConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KokkosSimdBackends"
  )
endif()
