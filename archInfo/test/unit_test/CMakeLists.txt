find_package(GTest QUIET)

if (NOT GTest_FOUND)
  include(FetchContent)

  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  list(APPEND CMAKE_MESSAGE_INDENT "[googletest] ")
  FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6910c9d9165801d8827d628cb72eb7ea9dd538c5
  DOWNLOAD_EXTRACT_TIMESTAMP true)

  FetchContent_MakeAvailable(googletest)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

add_executable(archInfoTest TestArchInfo.cpp)
target_link_libraries(archInfoTest PRIVATE GTest::gtest Kokkos::kokkos cexa::ArchInfo)

include(GoogleTest)
gtest_discover_tests(archInfoTest DISCOVERY_MODE PRE_TEST)
