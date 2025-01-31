find_package(googletest QUIET 1.11.0)
if(googletest_FOUND)
  message(STATUS "Using googletest found in ${googletest_DIR}")
else()
  message(STATUS "No installed googletest found, fetching from GitHub")
  include(FetchContent)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)

  list(APPEND CMAKE_MESSAGE_INDENT "[googletest] ")
  FetchContent_Declare(
    googletest
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
  )
  FetchContent_MakeAvailable(googletest)
  list(POP_BACK CMAKE_MESSAGE_INDENT)

  # Suppress clang-tidy diagnostics on code that we do not have control over
  if(CMAKE_CXX_CLANG_TIDY)
    set_target_properties(googletest PROPERTIES CXX_CLANG_TIDY "")
  endif()

  target_compile_options(gtest PRIVATE -w)
  target_compile_options(gtest_main PRIVATE -w)
endif()

add_executable(UnitTest UnitTestMain.cpp TestSIMD.cpp)
target_link_libraries(UnitTest PRIVATE math-simd GTest::gtest_main Kokkos::kokkoscore Kokkos::kokkossimd)
target_include_directories(UnitTest PRIVATE include)
target_compile_options(UnitTest PRIVATE -march=native $<$<CXX_COMPILER_ID:IntelLLVM>:-fp-model=precise>)

if (UNIX)
  target_link_libraries(UnitTest PRIVATE pthread)
  target_compile_options(UnitTest PRIVATE -pthread)
endif()

include(GoogleTest)
gtest_discover_tests(UnitTest)
