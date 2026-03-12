include(GNUInstallDirs)

add_library(archInfo)
add_library(cexa::ArchInfo ALIAS archInfo)

target_sources(archInfo PRIVATE cexa_ArchInfo.cpp)

target_include_directories(archInfo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(archInfo PUBLIC cxx_std_17)
target_link_libraries(archInfo PUBLIC Kokkos::kokkos)
if ("${Kokkos_ENABLE_CUDA}")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(archInfo PRIVATE CUDA::nvml)
endif()

install(
  TARGETS archInfo
  EXPORT CexaArchInfoTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES cexa_ArchInfo.hpp DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
