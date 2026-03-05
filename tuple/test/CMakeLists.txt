# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
find_package(GTest QUIET)
if (NOT GTest_FOUND)
  include(FetchContent)

  list(APPEND CMAKE_MESSAGE_INDENT "[googletest] ")
  FetchContent_Declare(
    googletest
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH MD5=b6f100bc2a5853a48046aa168ececf84
  )
  FetchContent_MakeAvailable(googletest)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

add_library(cexa.tuple.test_main)
target_sources(cexa.tuple.test_main PRIVATE main.cpp)
target_link_libraries(cexa.tuple.test_main PRIVATE Kokkos::kokkos GTest::gtest)

add_library(cexa.tuple.test INTERFACE)
target_include_directories(cexa.tuple.test INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(cexa.tuple.test INTERFACE cexa::tuple Kokkos::kokkos GTest::gtest)

include(GoogleTest)

if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
  set(CEXA_TUPLE_ENABLE_DEVICE_TESTS ON)
else()
  set(CEXA_TUPLE_ENABLE_DEVICE_TESTS OFF)
endif()

# This macro can be passed an additional argument which will be used to prefix
# the name of the test executable, as CMake doesn't allow multiple executables
# with the same name
macro(add_tuple_test FILE_NAME)
  set(FILENAME ${FILE_NAME})
  cmake_path(GET FILENAME STEM LAST_ONLY TEST_NAME)
  if(NOT "${ARGV1}" STREQUAL "")
    set(TEST_NAME "${ARGV1}.${TEST_NAME}")
  endif()
  add_executable(${TEST_NAME})
  target_sources(${TEST_NAME} PRIVATE ${FILE_NAME})
  target_link_libraries(${TEST_NAME} PRIVATE cexa.tuple.test cexa.tuple.test_main)
  target_compile_definitions(${TEST_NAME} PRIVATE TEST_EXECSPACE=Kokkos::DefaultHostExecutionSpace CEXA_IMPL_HOST_TEST)
  gtest_discover_tests(${TEST_NAME} DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 6000)
  if (CEXA_TUPLE_ENABLE_DEVICE_TESTS)
    set(DEVICE_TEST_NAME "${TEST_NAME}_device")
    add_executable(${DEVICE_TEST_NAME})
    target_sources(${DEVICE_TEST_NAME} PRIVATE ${FILE_NAME})
    target_link_libraries(${DEVICE_TEST_NAME} PRIVATE cexa.tuple.test cexa.tuple.test_main)
    target_compile_definitions(${DEVICE_TEST_NAME} PRIVATE TEST_EXECSPACE=Kokkos::DefaultExecutionSpace)
    gtest_discover_tests(${DEVICE_TEST_NAME} DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 6000)
  endif()
endmacro()

macro(add_tuple_compile_test FILE_NAME)
  set(FILENAME ${FILE_NAME})
  cmake_path(GET FILENAME STEM LAST_ONLY TEST_NAME)
  add_executable(${TEST_NAME})
  target_sources(${TEST_NAME} PRIVATE ${FILE_NAME})
  target_link_libraries(${TEST_NAME} PRIVATE cexa.tuple.test)
endmacro()

add_subdirectory(tuple.cnstr)
add_subdirectory(tuple.assign)
add_subdirectory(tuple.helper)
add_subdirectory(tuple.creation)
add_subdirectory(tuple.elem)
add_subdirectory(tuple.swap)
add_subdirectory(tuple.special)
add_subdirectory(tuple.general)
add_subdirectory(tuple.rel)
add_subdirectory(tuple.apply)
add_subdirectory(tuple.extra)
# add_subdirectory(tuple.traits)
