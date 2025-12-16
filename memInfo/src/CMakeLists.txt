add_library(memInfo INTERFACE)
target_include_directories(memInfo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(memInfo INTERFACE Kokkos::kokkos)
