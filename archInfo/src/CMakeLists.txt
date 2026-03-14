include(GNUInstallDirs)

add_library(archInfo)
add_library(cexa::ArchInfo ALIAS archInfo)

target_sources(
  archInfo
  PUBLIC
    FILE_SET HEADERS
    FILES
      cexa_ArchInfo.hpp
  PRIVATE
    cexa_ArchInfo.cpp
)

target_compile_features(archInfo PUBLIC cxx_std_17)
target_link_libraries(archInfo PUBLIC Kokkos::kokkos)
if ("${Kokkos_ENABLE_CUDA}")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(archInfo PRIVATE CUDA::nvml)
endif()

install(
  TARGETS archInfo
  EXPORT CexaArchInfoTargets
  FILE_SET HEADERS
)
