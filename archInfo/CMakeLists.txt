cmake_minimum_required(VERSION 3.21.1)

project(cexa-experimental-archInfo LANGUAGES CXX VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")
find_package(Kokkos REQUIRED)

option(CEXA_ARCHINFO_ENABLE_TESTS "Build the unit tests" OFF)

add_subdirectory(src)

if (CEXA_ARCHINFO_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test/unit_test)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  CexaArchInfoConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  EXPORT CexaArchInfoTargets
  FILE CexaArchInfoTargets.cmake
  NAMESPACE cexa::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaArchInfo"
)

install(FILES
  "${CMAKE_CURRENT_LIST_DIR}/cmake/CexaArchInfoConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/CexaArchInfoConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaArchInfo")

