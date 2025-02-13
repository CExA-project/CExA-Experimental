
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
  )
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(MemInfoTest TestMemInfo.cpp)
target_link_libraries(MemInfoTest GTest::gtest_main Kokkos::kokkos memInfo)

include(GoogleTest)
gtest_discover_tests(MemInfoTest)
