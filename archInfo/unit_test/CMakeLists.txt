
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6910c9d9165801d8827d628cb72eb7ea9dd538c5
  DOWNLOAD_EXTRACT_TIMESTAMP true)

FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(archInfoTest TestArchInfo.cpp)
target_link_libraries(archInfoTest GTest::gtest_main Kokkos::kokkos CexaExp::ArchInfo)

include(GoogleTest)
gtest_discover_tests(archInfoTest)
