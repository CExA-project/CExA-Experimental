# SPDX-FileCopyrightText: 2025 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
find_package(GTest QUIET)
if (NOT GTest_FOUND)
  FetchContent_Declare(GTest
    URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    DOWNLOAD_NO_PROGRESS TRUE)
  FetchContent_MakeAvailable(GTest)
endif()

enable_testing()

add_executable(accuracy_test accuracy.cpp)
target_link_libraries(accuracy_test PRIVATE Kokkos::kokkos simd-backends GTest::gtest)

include(GoogleTest)
gtest_discover_tests(accuracy_test)
