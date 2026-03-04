add_library(archInfo)
add_library(CexaExp::ArchInfo ALIAS archInfo)

target_sources(archInfo PRIVATE cexa_ArchInfo.cpp)
target_include_directories(archInfo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(archInfo PUBLIC cxx_std_20)
# target_compile_options(archInfo PRIVATE -Wall -Wextra)
target_link_libraries(archInfo PUBLIC Kokkos::kokkos)

install(TARGETS archInfo EXPORT CexaExpTargets)
install(FILES cexa_ArchInfo.hpp DESTINATION include)
