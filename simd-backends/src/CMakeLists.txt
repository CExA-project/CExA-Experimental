configure_file(
  CEXA_SIMD_Backends.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/CEXA_SIMD_Backends.hpp"
  @ONLY
)

add_library(simd-backends INTERFACE)
add_library(cexa::simd-backends ALIAS simd-backends)
target_sources(
  simd-backends
  INTERFACE
    FILE_SET HEADERS
    BASE_DIRS
      "${CMAKE_CURRENT_LIST_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}"
    FILES
      CEXA_SIMD_SLEEF.hpp
      CEXA_SIMD_SLEEF.hpp
      "${CMAKE_CURRENT_BINARY_DIR}/CEXA_SIMD_Backends.hpp"
)
target_link_libraries(simd-backends INTERFACE Kokkos::kokkossimd)

if(CEXA_SIMD_ENABLE_SLEEF)
  target_link_libraries(simd-backends INTERFACE sleef)
endif()
