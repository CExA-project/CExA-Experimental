# Find or download google/benchmark library
find_package(benchmark QUIET 1.5.6)
if(benchmark_FOUND)
  message(STATUS "Using google benchmark found in ${benchmark_DIR}")
else()
  message(STATUS "No installed google benchmark found, fetching from GitHub")
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF)

  list(APPEND CMAKE_MESSAGE_INDENT "[benchmark] ")
  FetchContent_Declare(
    googlebenchmark
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    URL https://github.com/google/benchmark/archive/refs/tags/v1.7.1.tar.gz
    URL_HASH MD5=0459a6c530df9851bee6504c3e37c2e7
  )
  FetchContent_MakeAvailable(googlebenchmark)
  list(POP_BACK CMAKE_MESSAGE_INDENT)

  # Suppress clang-tidy diagnostics on code that we do not have control over
  if(CMAKE_CXX_CLANG_TIDY)
    set_target_properties(benchmark PROPERTIES CXX_CLANG_TIDY "")
  endif()

  target_compile_options(benchmark PRIVATE -w)
  target_compile_options(benchmark_main PRIVATE -w)
endif()

add_executable(bench_kokkos SIMD_bench_kokkos.cpp)
target_link_libraries(bench_kokkos PRIVATE Kokkos::kokkos benchmark::benchmark math-simd)
target_compile_options(bench_kokkos PRIVATE -march=native -O3 -ftree-vectorize)
if (UNIX)
  target_link_libraries(bench_kokkos PRIVATE pthread)
  target_compile_options(bench_kokkos PRIVATE -pthread)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL IntelLLVM)
  add_executable(bench_intrinsics SIMD_bench_intrinsics.cpp)
  target_link_libraries(bench_intrinsics PRIVATE Kokkos::kokkos benchmark::benchmark math-simd)
  target_compile_options(bench_intrinsics PRIVATE -march=native -O3 -ftree-vectorize)
  if (UNIX)
    target_link_libraries(bench_intrinsics PRIVATE pthread)
    target_compile_options(bench_intrinsics PRIVATE -pthread)
  endif()
endif()
