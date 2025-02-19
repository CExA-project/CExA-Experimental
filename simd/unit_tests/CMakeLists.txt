# SPDX-FileCopyrightText: 2025 CExA-project
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

CPMAddPackage(
  googletest
  GITHUB_REPOSITORY google/googletest
  VERSION 1.15.2
  OPTIONS "BUILD_GMOCK OFF" "INSTALL_GTEST OFF"
)

add_executable(UnitTest UnitTestMain.cpp TestSIMD.cpp)
target_link_libraries(
  UnitTest
  PRIVATE math-simd GTest::gtest_main Kokkos::kokkoscore Kokkos::kokkossimd
)
target_include_directories(UnitTest PRIVATE include)

if("${COMPILER_SUPPORTS_MARCH_NATIVE}")
  target_compile_options(UnitTest PRIVATE -march=native)
endif()

if("${COMPILER_SUPPORTS_PTHREAD}")
  target_link_libraries(UnitTest PRIVATE pthread)
  target_compile_options(UnitTest PRIVATE -pthread)
endif()

if("${SIMD_USE_INTEL_FP_PRECISE}" AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL IntelLLVM)
  target_compile_options(UnitTest INTERFACE -fp-model=precise)
endif()

include(GoogleTest)
gtest_discover_tests(UnitTest)
