# set CMAKE_BUILD_TYPE if not defined
if(NOT CMAKE_BUILD_TYPE)
    set(default_build_type "RelWithDebInfo")
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(
        CMAKE_BUILD_TYPE
        "${default_build_type}"
        CACHE
        STRING
        "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
        FORCE
    )
endif()

# find Kokkos as an already existing target
if(TARGET Kokkos::kokkos)
    return()
endif()

# find Kokkos as installed
find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
    message(STATUS "Kokkos provided as installed: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
    message(STATUS "  With backend: ${Kokkos_DEVICES}")
    message(STATUS "  With options: ${Kokkos_OPTIONS}")
    message(STATUS "  For architectures: ${Kokkos_ARCH}")
    return()
endif()

# find Kokkos as an existing source directory
set(
    CexaExperimental_KOKKOS_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/kokkos"
    CACHE
    PATH
    "Path to the local source directory of Kokkos"
)
if(EXISTS "${CexaExperimental_KOKKOS_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "Kokkos provided as a source directory: ${CexaExperimental_KOKKOS_SOURCE_DIR}")

    add_subdirectory("${CexaExperimental_KOKKOS_SOURCE_DIR}" kokkos)
    set(Kokkos_FOUND True)

    return()
endif()

# download Kokkos from release and find it
message(STATUS "Kokkos downloaded: ${CexaExperimental_KOKKOS_SOURCE_DIR}")

include(FetchContent)

FetchContent_Declare(
    kokkos
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://github.com/kokkos/kokkos/releases/download/5.0.2/kokkos-5.0.2.tar.gz
    URL_HASH SHA256=188817bb452ca805ee8701f1c5adbbb4fb83dc8d1c50624566a18a719ba0fa5e
    SOURCE_DIR ${CexaExperimental_KOKKOS_SOURCE_DIR}
)
FetchContent_MakeAvailable(kokkos)
set(Kokkos_FOUND True)
