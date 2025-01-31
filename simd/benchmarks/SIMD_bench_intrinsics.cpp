#include <benchmark/benchmark.h>
#include <cassert>
#include <immintrin.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <numeric>
#include <type_traits>

#include <AVX2_Math.hpp>

#ifdef KOKKOS_ARCH_AVX512XEON
#include <AVX512_Math.hpp>
#endif

enum class Intrinsics {
    Intel,
    Custom,
};

template<std::size_t width, typename T>
struct VecType {};

template<>
struct VecType<4, float> {
    using value_type = __m128;
};

template<>
struct VecType<8, float> {
    using value_type = __m256;
};

template<>
struct VecType<4, double> {
    using value_type = __m256d;
};

template<>
struct VecType<16, float> {
    using value_type = __m512;
};

template<>
struct VecType<8, double> {
    using value_type = __m512d;
};

template<std::size_t width, typename T>
void load_vec(typename VecType<width, decltype(T{})>::value_type& vec, T* addr) {
    constexpr bool is_m128 = width == 4 && std::is_same_v<T, float>;
    constexpr bool is_m256 = width == 8 && std::is_same_v<T, float>;
    constexpr bool is_m512 = width == 16 && std::is_same_v<T, float>;
    constexpr bool is_m256d = width == 4 && std::is_same_v<T, double>;
    constexpr bool is_m512d = width == 8 && std::is_same_v<T, double>;

    if constexpr (is_m128) {
        vec = _mm_loadu_ps(addr);
    } else if constexpr (is_m256) {
        vec = _mm256_loadu_ps(addr);
    } else if constexpr (is_m256d) {
        vec = _mm256_loadu_pd(addr);
#ifdef KOKKOS_ARCH_AVX512XEON
    } else if constexpr (is_m512) {
        vec = _mm512_loadu_ps(addr);
    } else if constexpr (is_m512d) {
        vec = _mm512_loadu_pd(addr);
#endif
    }
}

template<std::size_t width, typename T, typename U>
void store_vec(U vec, T* addr) {
    constexpr bool is_m128 = width == 4 && std::is_same_v<T, float>;
    constexpr bool is_m256 = width == 8 && std::is_same_v<T, float>;
    constexpr bool is_m512 = width == 16 && std::is_same_v<T, float>;
    constexpr bool is_m256d = width == 4 && std::is_same_v<T, double>;
    constexpr bool is_m512d = width == 8 && std::is_same_v<T, double>;

    if constexpr (is_m128) {
        _mm_storeu_ps(addr, vec);
    } else if constexpr (is_m256) {
        _mm256_storeu_ps(addr, vec);
    } else if constexpr (is_m256d) {
        _mm256_storeu_pd(addr, vec);
#ifdef KOKKOS_ARCH_AVX512XEON
    } else if constexpr (is_m512) {
        _mm512_storeu_ps(addr, vec);
    } else if constexpr (is_m512d) {
        _mm512_storeu_pd(addr, vec);
#endif
    }
}

template<Intrinsics intrinsics, std::size_t width, typename T, typename U>
inline U tested_exp(U x) {
    constexpr bool is_m128 = width == 4 && std::is_same_v<T, float>;
    constexpr bool is_m256 = width == 8 && std::is_same_v<T, float>;
    constexpr bool is_m512 = width == 16 && std::is_same_v<T, float>;
    constexpr bool is_m256d = width == 4 && std::is_same_v<T, double>;
    constexpr bool is_m512d = width == 8 && std::is_same_v<T, double>;

    if constexpr (width == 1) {
        return Kokkos::exp(x);
    } else if constexpr (intrinsics == Intrinsics::Intel) {
        if constexpr (is_m128) {
            return _mm_exp_ps(x);
        } else if constexpr (is_m256) {
            return _mm256_exp_ps(x);
        } else if constexpr (is_m256d) {
            return _mm256_exp_pd(x);
#ifdef KOKKOS_ARCH_AVX512XEON
        } else if constexpr (is_m512) {
            return _mm512_exp_ps(x);
        } else if constexpr (is_m512d) {
            return _mm512_exp_pd(x);
#endif
        }
    } else {
        if constexpr (is_m128) {
            return avx2::exp4f(x);
        } else if constexpr (is_m256) {
            return avx2::exp8f(x);
        } else if constexpr (is_m256d) {
            return avx2::exp4d(x);
#ifdef KOKKOS_ARCH_AVX512XEON
        } else if constexpr (is_m512) {
            return avx512::exp16f(x);
        } else if constexpr (is_m512d) {
            return avx512::exp8d(x);
#endif
        }
    }
}

template<typename data_type>
data_type* setup(double lower_bound, double upper_bound, long samples) {
    static_assert(std::is_floating_point<data_type>::value);
    assert(samples > 0);
    assert(lower_bound < upper_bound);

    const double step = (upper_bound - lower_bound) / (double)samples;
    data_type* data_test = new data_type[samples];
    for (int i = 0; i < samples; i++) {
        data_test[i] = lower_bound + i * step;
    }
    return data_test;
}

template<typename data_type, std::size_t width, Intrinsics intrinsics>
static void bench_function(benchmark::State& state) {
    double lower_bound = -80.0;
    double upper_bound = 80.0;
    long samples = 1000000;

    assert(samples > 0);
    assert(lower_bound < upper_bound);

    double time = 0.0;
    const data_type* data_test = setup<data_type>(lower_bound, upper_bound, samples);
    data_type* result = new data_type[samples];

    for (auto _: state) {
        for (size_t i = 0; i < samples; i += width) {
            if constexpr (width == 1) {
                data_type res;
                benchmark::DoNotOptimize(
                    res = tested_exp<intrinsics, width, data_type>(data_test[i])
                );
                result[i] = res;
            } else {
                using simd_type = typename VecType<width, data_type>::value_type;
                simd_type v;
                load_vec<width>(v, &data_test[i]);
                simd_type res;
                benchmark::DoNotOptimize(
                    res = tested_exp<intrinsics, width, data_type>(v)
                );
                store_vec<width>(res, &result[i]);
            }
        }
    }

    delete[] data_test;
    delete[] result;
}

template<typename T>
T baseline;

template<typename T, std::size_t width>
T baseline_intel;

#define GENERATE_BENCHMARK(TYPE, WIDTH)                                    \
    BENCHMARK(bench_function<TYPE, WIDTH, Intrinsics::Intel>)              \
        ->Name(std::string("intel ") + #WIDTH + std::string(" ") + #TYPE)  \
        ->Iterations(32)                                                   \
        ->Repetitions(256)                                                 \
        ->ReportAggregatesOnly(true)                                       \
        ->ComputeStatistics(                                               \
            "speedup from scalar",                                         \
            [](const std::vector<double>& v) -> double {                   \
                double accum = std::accumulate(v.begin(), v.end(), 0.0);   \
                double mean = accum / v.size();                            \
                baseline_intel<TYPE, WIDTH> = mean;                        \
                return baseline<TYPE> / mean;                              \
            },                                                             \
            benchmark::StatisticUnit::kPercentage                          \
        )                                                                  \
        ->Unit(benchmark::kMillisecond);                                   \
                                                                           \
    BENCHMARK(bench_function<TYPE, WIDTH, Intrinsics::Custom>)             \
        ->Name(std::string("custom ") + #WIDTH + std::string(" ") + #TYPE) \
        ->Iterations(32)                                                   \
        ->Repetitions(256)                                                 \
        ->ReportAggregatesOnly(true)                                       \
        ->ComputeStatistics(                                               \
            "speedup from scalar",                                         \
            [](const std::vector<double>& v) -> double {                   \
                double accum = std::accumulate(v.begin(), v.end(), 0.0);   \
                double mean = accum / v.size();                            \
                return baseline<TYPE> / mean;                              \
            },                                                             \
            benchmark::StatisticUnit::kPercentage                          \
        )                                                                  \
        ->ComputeStatistics(                                               \
            "speedup from intel",                                          \
            [](const std::vector<double>& v) -> double {                   \
                double accum = std::accumulate(v.begin(), v.end(), 0.0);   \
                double mean = accum / v.size();                            \
                return baseline_intel<TYPE, WIDTH> / mean;                 \
            },                                                             \
            benchmark::StatisticUnit::kPercentage                          \
        )                                                                  \
        ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<float, 1, Intrinsics::Intel>)
    ->Name("Sequential float")
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics(
        "speedup",
        [](const std::vector<double>& v) -> double {
            double accum = std::accumulate(v.begin(), v.end(), 0.0);
            baseline<float> = accum / v.size();
            return 1.0;
        },
        benchmark::StatisticUnit::kPercentage
    )
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<double, 1, Intrinsics::Intel>)
    ->Name("Sequential double")
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics(
        "speedup",
        [](const std::vector<double>& v) -> double {
            double accum = std::accumulate(v.begin(), v.end(), 0.0);
            baseline<double> = accum / v.size();
            return 1.0;
        },
        benchmark::StatisticUnit::kPercentage
    )
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
