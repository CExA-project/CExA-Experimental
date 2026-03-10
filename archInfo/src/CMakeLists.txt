add_library(archInfo)
add_library(cexa::ArchInfo ALIAS archInfo)

cmake_host_system_information(RESULT CEXA_DISTRIB_TYPE    QUERY OS_NAME)
cmake_host_system_information(RESULT CEXA_DISTRIB_NAME    QUERY DISTRIB_PRETTY_NAME)
cmake_host_system_information(RESULT CEXA_DISTRIB_VERSION QUERY OS_VERSION)

cmake_host_system_information(RESULT CEXA_PROCESSOR_NAME QUERY PROCESSOR_NAME)
cmake_host_system_information(RESULT CEXA_CORE_COUNT     QUERY NUMBER_OF_PHYSICAL_CORES)
cmake_host_system_information(RESULT CEXA_THREAD_COUNT   QUERY NUMBER_OF_LOGICAL_CORES)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cexa_compilInfo.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/cexa_compilInfo.hpp
  @ONLY
)

target_sources(archInfo PRIVATE cexa_ArchInfo.cpp)

target_include_directories(archInfo PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_features(archInfo PUBLIC cxx_std_20)
target_compile_options(archInfo PRIVATE -Wall -Wextra)
target_link_libraries(archInfo PUBLIC Kokkos::kokkos)

install(TARGETS archInfo EXPORT CexaExpTargets)
install(FILES cexa_ArchInfo.hpp DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cexa_compilInfo.hpp DESTINATION include)
