#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <cassert>
#include <immintrin.h>
#include <type_traits>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <AVX2_Math.hpp>

struct BenchmarkResult {
    double time;
    double speedup;
    char name[32];
};

template<typename simd_type>
constexpr auto tested_exp(simd_type x) {
    if constexpr (std::is_same_v<simd_type, __m256d>) {
        #ifdef __INTEL_BENCHMARK
            return _mm256_exp_pd(x);
        #else
            return exp4d(x);
        #endif
    } else if constexpr (std::is_same_v<simd_type, __m256>) {
        #ifdef __INTEL_BENCHMARK
            return _mm256_exp_ps(x);
        #else
            return exp8f(x);
        #endif
    } else if constexpr (std::is_same_v<simd_type, __m128>) {
        #ifdef __INTEL_BENCHMARK
            return _mm_exp_ps(x);
        #else
            return exp4f(x);
        #endif
    } else if constexpr (std::is_same_v<simd_type, double> || 
                         std::is_same_v<simd_type, float>  || 
                         std::is_same_v<simd_type, int>) {
        return std::exp(x);
    } else {
        static_assert(false, "unsupported type");
    }
}

template<typename T, std::size_t width>
constexpr auto init_test(T x) {
    if constexpr (width == 4) {
        if constexpr (std::is_same_v<T, float>) {
            return __m128{x, x, x, x};
        } else {
            return __m256d{x, x, x, x};
        }
    } else if constexpr (width == 8 && std::is_same_v<T, float>) {
        return __m256{x, x, x, x, x, x, x, x};
    } else if constexpr (width == 1 && std::is_same_v<T, double> ||
                         width == 1 && std::is_same_v<T, float>  ||
                         width == 1 && std::is_same_v<T, int>) {
        return x;
    }else {
        static_assert(false, "Only floating point vector types are supported");
    }
}

template<typename T>
constexpr auto init_test(T x0, T x1, T x2, T x3) {
    if constexpr (std::is_same_v<T, float>) {
        return __m128{x0, x1, x2, x3};
    } else if constexpr (std::is_same_v<T, double>){
        return __m256d{x0, x1, x2, x3};
    } else {
        static_assert(false, "Only double and float for 4 lanes are supported");
    }
}

template<typename T>
constexpr auto init_test(T x0, T x1, T x2, T x3, T x4, T x5, T x6, T x7) {
    if constexpr (std::is_same_v<T, float>) {
        return __m256{x0, x1, x2, x3, x4, x5, x6, x7};
    } else {
        static_assert(false, "Only float for 8 lanes are supported");
    }
}

void print_result(const std::vector<BenchmarkResult> results, long sample){
    char* fileOutput = (char*)malloc(sizeof(char) * 256);
    char date[32];
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::strftime(date, sizeof(date), "%d-%m-%Y-%H-%M-%S", std::localtime(&in_time_t));
    
    #ifdef __INTEL_BENCHMARK
        snprintf(fileOutput, 256, "exp_benchmark_icpx_%s", date);
    #else
        snprintf(fileOutput, 256, "exp_benchmark_%s", date);
    #endif
    FILE* ofile = fopen(fileOutput, "w");
    
    fprintf(ofile, "%-16s %10s %12s %12s\n", "Fonction", "Sample", "Time", "Speedup");
    for (const auto& result : results) {
        fprintf(ofile, "%-16s %10ld %12.4f %12.4f\n", result.name, sample, result.time, result.speedup);
    }
    fclose(ofile);
    free(fileOutput);
}

void print_usage(const char* argv0 ) {
    printf("%s\n", argv0);
    printf("Arguments: S N L U\n");
    printf("  S:   Number of samples\n");
    printf("  N:   Repeat the experiment N times\n");
    printf("  L:   Lower bound of the range\n");
    printf("  U:   Upper bound of the range\n");
    printf("Example Arguments:\n");
}

template <typename data_type, std::size_t width>
double bench_function(long samples, long nrepeat) {
    Kokkos::Timer timer;
    long simd_loop = static_cast<long>((double)samples/(double)width);
    long rest = samples - simd_loop*width;
    volatile data_type data_simd[width];
    double time = 0.0;
    for (int meta_rep = 0; meta_rep < nrepeat; meta_rep++) {
        timer.reset();
        for (int i = 0; i < simd_loop; i++)
        { 
            auto v = init_test<data_type, width>(i);
            volatile auto res = tested_exp(v);
            if constexpr (width != 1) {
                for (int lane = 0; lane < width; ++lane) {
                    data_simd[lane] = res[lane];
                }
            }
        }
        time += timer.seconds();
    }
    time = time/static_cast<double>(nrepeat);
    return time;
}

template <typename data_type, std::size_t width>
double bench_function(double lower_bound, double upper_bound, long samples, long nrepeat) {
    Kokkos::Timer timer;
    const double step = (upper_bound - lower_bound)/(double)samples;
    const long simd_loop = static_cast<long>((double)samples/(double)width);
    volatile data_type data_simd[width];
    double time = 0.0;

    for (int meta_rep = 0; meta_rep < nrepeat; meta_rep++) {
        timer.reset();
        for (int i = 0; i < simd_loop; i++)
        {
            volatile auto x = lower_bound + i*step;
            if constexpr (width == 1) {
                volatile auto res = tested_exp(x);
                data_simd[0] = res;
            } else if constexpr (width == 4) {
                auto v = init_test<data_type>(x, x+step, x+2*step, x+3*step);
                volatile auto res = tested_exp(v);
                for (int lane = 0; lane < width; ++lane) {
                    data_simd[lane] = res[lane];
                }
            } else if constexpr (width == 8) {
                auto v = init_test<data_type>(x, x+step, x+2*step, x+3*step, x+4*step, x+5*step, x+6*step, x+7*step);
                volatile auto res = tested_exp(v);
                for (int lane = 0; lane < width; ++lane) {
                    data_simd[lane] = res[lane];
                }
            }
        }
        time += timer.seconds();
    }
    time = time/static_cast<double>(nrepeat);
    return time;
}

int main (int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    long samples = std::stol(argv[1]);
    long nrepeat = std::stol(argv[2]); 
    long lower_bound = std::stod(argv[3]);
    long upper_bound = std::stod(argv[4]);
    assert(samples > 0);
    assert(nrepeat > 0);
    
    double elapsed_time = 0.0;
    std::vector<BenchmarkResult> results;
    Kokkos::initialize();
    {
        // fp32 reference
        double base_time_fp32 = bench_function<float,1>(lower_bound, upper_bound, samples, nrepeat);
        results.push_back({base_time_fp32, base_time_fp32/base_time_fp32 ,"expf"});

        // fp64 reference
        double base_time_fp64 = bench_function<double,1>(lower_bound, upper_bound, samples, nrepeat);
        results.push_back({base_time_fp64, base_time_fp64/base_time_fp64 ,"expd"});

        // 4 lanes double
        elapsed_time = bench_function<double,4>(lower_bound, upper_bound, samples, nrepeat);
        results.push_back({elapsed_time, base_time_fp64/elapsed_time, "exp4d"});              

        // 8 lanes float
        elapsed_time = bench_function<float,8>(lower_bound, upper_bound, samples, nrepeat);
        results.push_back({elapsed_time, base_time_fp32/elapsed_time, "exp8f"});
        
        // 4 lanes float
        elapsed_time = bench_function<float,4>(lower_bound, upper_bound, samples, nrepeat);
        results.push_back({elapsed_time, base_time_fp32/elapsed_time, "exp4f"});
    }
    print_result(results, samples);
    Kokkos::finalize();
}
