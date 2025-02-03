#include <benchmark/benchmark.h>
#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <new>
#include <numeric>
#include <type_traits>

#include <AVX2_Math.hpp>
#include <xmmintrin.h>

#ifdef KOKKOS_ARCH_AVX512XEON
#include <AVX512_Math.hpp>
#endif

enum class Intrinsics {
    Intel,
    Custom,
};

template<Intrinsics intrinsics>
__m128 tested_exp(__m128 x) {
    if constexpr (intrinsics == Intrinsics::Intel) {
        return _mm_exp_ps(x);
    } else {
        return avx2::exp4f(x);
    }
}

template<Intrinsics intrinsics>
__m256d tested_exp(__m256d x) {
    if constexpr (intrinsics == Intrinsics::Intel) {
        return _mm256_exp_pd(x);
    } else {
        return avx2::exp4d(x);
    }
}

#ifndef KOKKOS_ARCH_AVX512XEON
template<Intrinsics intrinsics>
__m256 tested_exp(__m256 x) {
    if constexpr (intrinsics == Intrinsics::Intel) {
        return _mm256_exp_ps(x);
    } else {
        return avx2::exp8f(x);
    }
}
#else
template<Intrinsics intrinsics>
__m256 tested_exp(__m256 x) {
    if constexpr (intrinsics == Intrinsics::Intel) {
        return _mm256_exp_ps(x);
    } else {
        return avx512::exp8f(x);
    }
}

template<Intrinsics intrinsics>
__m512 tested_exp(__m512 x) {
    if constexpr (intrinsics == Intrinsics::Intel) {
        return _mm512_exp_ps(x);
    } else {
        return avx512::exp16f(x);
    }
}

template<Intrinsics intrinsics>
__m512d tested_exp(__m512d x) {
    if constexpr (intrinsics == Intrinsics::Intel) {
        return _mm512_exp_pd(x);
    } else {
        return avx512::exp8d(x);
    }
}
#endif

template<typename data_type>
void setup(data_type* data, std::size_t samples) {
    static_assert(std::is_floating_point<data_type>::value);
    double lower_bound = -80.0;
    double upper_bound = 80.0;

    assert(samples > 0);
    assert(lower_bound < upper_bound);

    const double step = (upper_bound - lower_bound) / (double)samples;
    for (int i = 0; i < samples; i++) {
        data[i] = lower_bound + i * step;
    }
}

template<typename data_type, std::size_t width, Intrinsics intrinsics>
static void bench_function(benchmark::State& state) {
    std::size_t samples = 1000000;

    data_type* data_test =
        new (std::align_val_t(width * sizeof(data_type))) data_type[samples];
    setup<data_type>(data_test, samples);
    data_type* result =
        new (std::align_val_t(width * sizeof(data_type))) data_type[samples];

    for (auto _: state) {
        for (std::size_t i = 0; i < samples; i += width) {
            if constexpr (width == 1) {
                benchmark::DoNotOptimize(result[i] = std::exp(data_test[i]));
            } else if constexpr (width == 4 && std::is_same_v<data_type, float>) {
                __m128 v = _mm_load_ps(&data_test[i]);
                __m128 res;
                benchmark::DoNotOptimize(res = tested_exp<intrinsics>(res));
                _mm_store_ps(&result[i], res);
            } else if constexpr (width == 8 && std::is_same_v<data_type, float>) {
                __m256 v = _mm256_load_ps(&data_test[i]);
                __m256 res;
                benchmark::DoNotOptimize(res = tested_exp<intrinsics>(res));
                _mm256_store_ps(&result[i], res);
            } else if constexpr (width == 4 && std::is_same_v<data_type, double>) {
                __m256d v = _mm256_load_pd(&data_test[i]);
                __m256d res;
                benchmark::DoNotOptimize(res = tested_exp<intrinsics>(res));
                _mm256_store_pd(&result[i], res);
#ifdef KOKKOS_ARCH_AVX512XEON
            } else if constexpr (width == 16 && std::is_same_v<data_type, float>) {
                __m512 v = _mm512_load_ps(&data_test[i]);
                __m512 res;
                benchmark::DoNotOptimize(res = tested_exp<intrinsics>(res));
                _mm512_store_ps(&result[i], res);
            } else if constexpr (width == 8 && std::is_same_v<data_type, double>) {
                __m512d v = _mm512_load_pd(&data_test[i]);
                __m512d res;
                benchmark::DoNotOptimize(res = tested_exp<intrinsics>(res));
                _mm512_store_pd(&result[i], res);
#endif
            }
        }
    }

    delete[] data_test;
    delete[] result;
}

#define GENERATE_BENCHMARK(TYPE, WIDTH)                                    \
    BENCHMARK(bench_function<TYPE, WIDTH, Intrinsics::Intel>)              \
        ->Name(std::string("intel ") + #WIDTH + std::string(" ") + #TYPE)  \
        ->ReportAggregatesOnly(true)                                       \
        ->Unit(benchmark::kMillisecond);                                   \
                                                                           \
    BENCHMARK(bench_function<TYPE, WIDTH, Intrinsics::Custom>)             \
        ->Name(std::string("custom ") + #WIDTH + std::string(" ") + #TYPE) \
        ->ReportAggregatesOnly(true)                                       \
        ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<float, 1, Intrinsics::Intel>)
    ->Name("Sequential float")
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<double, 1, Intrinsics::Intel>)
    ->Name("Sequential double")
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);

#ifdef KOKKOS_ARCH_AVX2
GENERATE_BENCHMARK(float, 4)
GENERATE_BENCHMARK(float, 8)
GENERATE_BENCHMARK(double, 4)
#endif

#ifdef KOKKOS_ARCH_AVX512XEON
GENERATE_BENCHMARK(float, 8)
GENERATE_BENCHMARK(float, 16)
GENERATE_BENCHMARK(double, 8)
#endif

BENCHMARK_MAIN();
