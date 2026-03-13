cmake_minimum_required(VERSION 3.23)

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
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  cmake/CexaArchInfoConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/CexaArchInfoConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaArchInfo"
)

install(
  EXPORT CexaArchInfoTargets
  FILE CexaArchInfoTargets.cmake
  NAMESPACE cexa::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaArchInfo"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/CexaArchInfoConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/CexaArchInfoConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CexaArchInfo")

