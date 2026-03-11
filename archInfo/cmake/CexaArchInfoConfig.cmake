include(CMakeFindDependencyMacro)

find_dependency(Kokkos REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/CexaArchInfoTargets.cmake")
