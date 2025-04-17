cmake_minimum_required(VERSION 3.16)

project(simd-backends)

find_package(Kokkos 4.5 REQUIRED CONFIG)

option(ENABLE_SLEEF "Use the sleef library to implement the simd math functions" OFF)
option(ENABLE_SVML "Use the svml to implement the simd math functions" OFF)
option(ENABLE_TESTS "Enable accuracy tests" OFF)

include(FetchContent)

if (ENABLE_SLEEF)
  find_package(sleef 3.6 QUIET CONFIG)
  if (NOT sleef_FOUND)
    set(SLEEF_BUILD_GNUABI_LIBS OFF)
    set(SLEEF_BUILD_TESTS OFF)
    FetchContent_Declare(sleef
      URL https://github.com/shibatch/sleef/archive/refs/tags/3.9.0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      DOWNLOAD_NO_PROGRESS TRUE)
    FetchContent_MakeAvailable(sleef)
  endif()
endif()

if (ENABLE_SVML AND NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL IntelLLVM)
  message(FATAL_ERROR "The SVML simd backend is only available with Intel compilers")
endif()

add_library(simd-backends INTERFACE)
target_include_directories(simd-backends INTERFACE ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(simd-backends INTERFACE Kokkos::kokkossimd)

if (ENABLE_SLEEF)
  target_link_libraries(simd-backends INTERFACE sleef)
  target_compile_definitions(simd-backends INTERFACE KOKKOS_ENABLE_SLEEF)
elseif(ENABLE_SVML)
  target_compile_definitions(simd-backends INTERFACE KOKKOS_ENABLE_SVML)
endif()

if (ENABLE_TESTS)
  add_subdirectory(test)
endif()
