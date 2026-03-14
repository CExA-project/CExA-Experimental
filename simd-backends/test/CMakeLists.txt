# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

find_package(GTest QUIET)
if (NOT GTest_FOUND)
  include(FetchContent)

  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  list(APPEND CMAKE_MESSAGE_INDENT "[googletest] ")

  FetchContent_Declare(GTest
    URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    DOWNLOAD_NO_PROGRESS TRUE)
  FetchContent_MakeAvailable(GTest)

  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

enable_testing()

add_executable(accuracy_test accuracy.cpp)
target_link_libraries(accuracy_test PRIVATE Kokkos::kokkos simd-backends GTest::gtest)

include(GoogleTest)
gtest_discover_tests(accuracy_test DISCOVERY_MODE PRE_TEST)
