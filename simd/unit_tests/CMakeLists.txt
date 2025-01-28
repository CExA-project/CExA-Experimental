include(CPM)

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.11.0
  OPTIONS "BUILD_GMOCK OFF" "INSTALL_GTEST OFF"
)

add_executable(UnitTest UnitTestMain.cpp TestSIMD.cpp)
target_link_libraries(UnitTest PRIVATE math-simd GTest::gtest_main Kokkos::kokkoscore Kokkos::kokkossimd)
target_include_directories(UnitTest PRIVATE include)
target_compile_options(UnitTest PRIVATE -march=native)

include(GoogleTest)
gtest_discover_tests(UnitTest)
