add_library(archInfo)
add_library(cexa::ArchInfo ALIAS archInfo)

target_sources(archInfo PRIVATE cexa_ArchInfo.cpp)

target_include_directories(archInfo PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_features(archInfo PUBLIC cxx_std_20)
target_compile_options(archInfo PRIVATE -Wall -Wextra)
target_link_libraries(archInfo PUBLIC Kokkos::kokkos)
if ("${Kokkos_ENABLE_CUDA}")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(archInfo PRIVATE CUDA::nvml)
endif()

install(TARGETS archInfo EXPORT CexaExpTargets)
install(FILES cexa_ArchInfo.hpp DESTINATION include)
