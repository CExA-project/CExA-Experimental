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


find_program(ICX_COMPILER icpx)
if(NOT ICX_COMPILER)
  message(ERROR "Intel C++ Compiler not found")
else()
  set(CMAKE_CXX_COMPILER icpx)
  set(BENCHMARK_NAME exp_bench_icpx)
  add_executable(exp_bench_icpx SIMD_bench.cpp)
  target_link_libraries(exp_bench_icpx Kokkos::kokkos benchmark::benchmark math-simd)
  target_compile_options(exp_bench_icpx PRIVATE -march=native -O3 -ftree-vectorize)
endif()
