#include <cassert>
#include <immintrin.h>
#include <type_traits>
#include <numeric>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <AVX2_Math.hpp>
#include <benchmark/benchmark.h>


double base_line_double = 0.0;
double base_line_float  = 0.0;

enum class Intrinsics {
    Intel,
    Custom,
};

template<Intrinsics intrinsics>
constexpr __m128 tested_exp(__m128 x) {
    if constexpr(intrinsics == Intrinsics::Intel) {
        return _mm_exp_ps(x);
    } else {
        return exp4f(x);
    }
}

template<Intrinsics intrinsics>
constexpr __m256 tested_exp(__m256 x) {
    if constexpr(intrinsics == Intrinsics::Intel) {
        return _mm256_exp_ps(x);
    } else {
        return exp8f(x);
    }
}

template<Intrinsics intrinsics>
constexpr __m256d tested_exp(__m256d x) {
    if constexpr(intrinsics == Intrinsics::Intel) {
        return _mm256_exp_pd(x);
    } else {
        return exp4d(x);
    }
}

template<typename T>
constexpr T tested_exp(T x) {
    return Kokkos::exp(x);
}

template<typename T, std::size_t width>
constexpr auto init_test(T x) {
    constexpr bool is_m128 = width == 4 && std::is_same_v<T, float>;
    constexpr bool is_m256 = width == 8 && std::is_same_v<T, float>;
    constexpr bool is_m256d = width == 4 && std::is_same_v<T, double>;
    static_assert(is_m128 || is_m256 || is_m256d, "only floating point vector types are supported");

    if constexpr(is_m128) {
        return __m128{x, x, x, x};
    } else if constexpr(is_m256d) {
        return __m256d{x, x, x, x};
    } else if constexpr(is_m256) {
        return __m256{x, x, x, x, x, x, x, x};
    } else {
        return x;
    }
}

template<typename T>
constexpr auto init_test(T x0, T x1, T x2, T x3) {
    constexpr bool is_m128 = std::is_same_v<T, float>;
    constexpr bool is_m256d = std::is_same_v<T, double>;
    static_assert(is_m128 || is_m256d, "Only double and float for 4 lanes are supported");

    if constexpr (is_m128) {
        return __m128{x0, x1, x2, x3};
    } else if constexpr (is_m256d) {
        return __m256d{x0, x1, x2, x3};
    }
}

template<typename T>
constexpr auto init_test(T x0, T x1, T x2, T x3, T x4, T x5, T x6, T x7) {
    constexpr bool is_m256 = std::is_same_v<T, float>;
    static_assert(is_m256, "Only float for 8 lanes are supported");
    if constexpr (is_m256) {
        return __m256{x0, x1, x2, x3, x4, x5, x6, x7};
    }
}

template <typename data_type>
data_type* setup(double lower_bound, double upper_bound, long samples) {
    static_assert(std::is_floating_point<data_type>::value);
    assert(samples > 0);
    assert(lower_bound < upper_bound);

    const double step = (upper_bound - lower_bound)/(double)samples;
    data_type* data_test = new data_type[samples];
    for (int i = 0; i < samples; i++) {
        data_test[i] = lower_bound + i*step;
    }
    return data_test;
}

template <typename data_type, std::size_t width, Intrinsics intrinsics>
static void bench_function(benchmark::State& state) {
    double lower_bound = state.range(0);
    double upper_bound = state.range(1);
    long samples = state.range(2);

    assert(samples > 0);
    assert(nrepeat > 0);
    assert(lower_bound < upper_bound);

    double time = 0.0;
    const volatile data_type* data_test = setup<data_type>(lower_bound, upper_bound, samples);
    volatile data_type* result = new data_type[samples];
    for (auto _ : state) {
        for (size_t i = 0; i < samples; i += width)
        {
            if constexpr (width == 1) {
                volatile auto res = tested_exp<data_type>(data_test[i]);
                result[i] = res;
            } else if constexpr (width == 4) {
                const auto v = init_test<data_type>(data_test[i], data_test[i+1], data_test[i+2], data_test[i+3]);
                volatile auto res = tested_exp<intrinsics>(v);
                for (size_t lane = 0; lane < width; ++lane) {
                    result[i+lane] = res[i+lane];
                }
            } else if constexpr (width == 8) {
                const auto v = init_test<data_type>(data_test[i], data_test[i+1], data_test[i+2], data_test[i+3], 
                                              data_test[i+4], data_test[i+5], data_test[i+6], data_test[i+7]);
                volatile auto res = tested_exp<intrinsics>(v);
                for (size_t lane = 0; lane < width; ++lane) {
                    result[i+lane] = res[i+lane];
                }
            }
        }
    }
    delete[] data_test;
    delete[] result;
    return;
}


BENCHMARK(bench_function<double,1, Intrinsics::Custom>)
    ->Name("exp1d")
    ->Args({-1000, 1000, 100000})
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics("speedUp", [](const std::vector<double>& v) -> double {
        double accum =  std::accumulate(v.begin(), v.end(), 0.0);
        base_line_double = accum / v.size();
        return 1.0;
    }, benchmark::StatisticUnit::kPercentage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<float,1, Intrinsics::Custom>)
    ->Name("exp1f")
    ->Args({-1000, 1000, 100000})
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics("speedUp", [](const std::vector<double>& v) -> double {
        double accum =  std::accumulate(v.begin(), v.end(), 0.0);
        base_line_float = accum / v.size();
        return 1.0;
    }, benchmark::StatisticUnit::kPercentage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<double,4, Intrinsics::Custom>)
    ->Name("exp4d-custom")
    ->Args({-1000, 1000, 100000})
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics("speedUp", [](const std::vector<double>& v) -> double {
        double accum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = accum / v.size();
        return base_line_double / mean;
    }, benchmark::StatisticUnit::kPercentage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<double,4, Intrinsics::Intel>)
    ->Name("exp4d-Intel")
    ->Args({-1000, 1000, 100000})
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics("speedUp", [](const std::vector<double>& v) -> double {
        double accum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = accum / v.size();
        return base_line_double / mean;
    }, benchmark::StatisticUnit::kPercentage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<float,8, Intrinsics::Custom>)
    ->Name("exp8f-custom")
    ->Args({-1000, 1000, 100000})
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics("speedUp", [](const std::vector<double>& v) -> double {
        double accum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = accum / v.size();
        return base_line_float / mean;
    }, benchmark::StatisticUnit::kPercentage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<float,8, Intrinsics::Intel>)
    ->Name("exp8f-Intel")
    ->Args({-1000, 1000, 100000})
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics("speedUp", [](const std::vector<double>& v) -> double {
        double accum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = accum / v.size();
        return base_line_float / mean;
    }, benchmark::StatisticUnit::kPercentage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<float,4, Intrinsics::Custom>)
    ->Name("exp4f-custom")
    ->Args({-1000, 1000, 100000})
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics("speedUp", [](const std::vector<double>& v) -> double {
        double accum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = accum / v.size();
        return base_line_float / mean;
    }, benchmark::StatisticUnit::kPercentage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<float,4, Intrinsics::Intel>)
    ->Name("exp4f-Intel")
    ->Args({-1000, 1000, 100000})
    ->Iterations(32)
    ->Repetitions(256)
    ->ReportAggregatesOnly(true)
    ->ComputeStatistics("speedUp", [](const std::vector<double>& v) -> double {
        double accum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = accum / v.size();
        return base_line_float / mean;
    }, benchmark::StatisticUnit::kPercentage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
