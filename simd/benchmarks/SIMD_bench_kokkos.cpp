#include <benchmark/benchmark.h>
#include <cassert>
#include <immintrin.h>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <numeric>
#include <type_traits>

enum class Intrinsics {
    Kokkos,
    Custom,
};

#ifdef KOKKOS_ARCH_AVX2
#include <AVX2_Math.hpp>
#endif

#ifdef KOKKOS_ARCH_AVX512XEON
#include <AVX512_Math.hpp>
#endif

template<typename Abi, typename DataType>
const char* simd_name = "unknown abi";
template<>
const char* simd_name<Kokkos::Experimental::simd_abi::scalar, float> = "scalar float";
template<>
const char* simd_name<Kokkos::Experimental::simd_abi::scalar, double> = "scalar double";

#ifdef KOKKOS_ARCH_AVX2
[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<double, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>
    custom_exp(Kokkos::Experimental::basic_simd<
               double,
               Kokkos::Experimental::simd_abi::avx2_fixed_size<4>> const& x) {
    return Kokkos::Experimental::
        basic_simd<double, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>(
            avx2::exp4d(static_cast<__m256d>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<8>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx2_fixed_size<8>> const& x) {
    return Kokkos::Experimental::
        basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<8>>(
            avx2::exp8f(static_cast<__m256>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx2_fixed_size<4>> const& x) {
    return Kokkos::Experimental::
        basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>(
            avx2::exp4f(static_cast<__m128>(x))
        );
}

template<>
const char* simd_name<Kokkos::Experimental::simd_abi::avx2_fixed_size<4>, float> =
    "AVX2 4 float";
template<>
const char* simd_name<Kokkos::Experimental::simd_abi::avx2_fixed_size<8>, float> =
    "AVX2 8 float";
template<>
const char* simd_name<Kokkos::Experimental::simd_abi::avx2_fixed_size<4>, double> =
    "AVX2 4 double";

#endif

#ifdef KOKKOS_ARCH_AVX512XEON
[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<double, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>
    custom_exp(Kokkos::Experimental::basic_simd<
               double,
               Kokkos::Experimental::simd_abi::avx512_fixed_size<8>> const& x) {
    return Kokkos::Experimental::
        basic_simd<double, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>(
            avx512::exp8d(static_cast<__m512d>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx512_fixed_size<16>> const& x) {
    return Kokkos::Experimental::
        basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>>(
            avx512::exp16f(static_cast<__m512>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx512_fixed_size<8>> const& x) {
    return Kokkos::Experimental::
        basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>(
            avx512::exp8f(static_cast<__m256>(x))
        );
}

template<>
const char* simd_name<Kokkos::Experimental::simd_abi::avx512_fixed_size<8>, float> =
    "AVX512 8 float";
template<>
const char* simd_name<Kokkos::Experimental::simd_abi::avx512_fixed_size<16>, float> =
    "AVX512 16 float";
template<>
const char* simd_name<Kokkos::Experimental::simd_abi::avx512_fixed_size<8>, double> =
    "AVX512 8 double";

#endif

template<Intrinsics intrinsics, typename Abi, typename data_type>
constexpr Kokkos::Experimental::basic_simd<data_type, Abi>
tested_exp(Kokkos::Experimental::basic_simd<data_type, Abi> const& x) {
    if constexpr (intrinsics == Intrinsics::Kokkos) {
        return Kokkos::exp(x);
    } else {
        return custom_exp(x);
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

template<typename Abi, typename data_type, Intrinsics intrinsics>
static void bench_function(benchmark::State& state) {
    double lower_bound = -80.0;
    double upper_bound = 80.0;
    long samples = 1000000;

    assert(samples > 0);
    assert(lower_bound < upper_bound);

    using simd_type = Kokkos::Experimental::basic_simd<data_type, Abi>;
    constexpr std::size_t width = simd_type::size();

    const data_type* data_test = setup<data_type>(lower_bound, upper_bound, samples);
    data_type* result = new data_type[samples];

    for (auto _: state) {
        for (size_t i = 0; i < samples; i += width) {
            simd_type v;
            v.copy_from(&data_test[i], Kokkos::Experimental::simd_flag_default);
            simd_type res;
            benchmark::DoNotOptimize(res = tested_exp<intrinsics>(v));
            res.copy_to(&result[i], Kokkos::Experimental::simd_flag_default);
        }
    }

    benchmark::DoNotOptimize(result[state.bytes_processed() % samples]);

    delete[] data_test;
    delete[] result;
}

template<typename T>
T baseline;

template<typename T, typename Abi>
T baseline_kokkos;

#define GENERATE_BENCHMARK(TYPE, ABI)                                    \
    BENCHMARK(bench_function<ABI, TYPE, Intrinsics::Kokkos>)             \
        ->Name(std::string("kokkos ") + simd_name<ABI, TYPE>)            \
        ->Iterations(32)                                                 \
        ->Repetitions(256)                                               \
        ->ReportAggregatesOnly(true)                                     \
        ->ComputeStatistics(                                             \
            "speedup from scalar",                                       \
            [](const std::vector<double>& v) -> double {                 \
                double accum = std::accumulate(v.begin(), v.end(), 0.0); \
                double mean = accum / v.size();                          \
                baseline_kokkos<TYPE, ABI> = mean;                       \
                return baseline<TYPE> / mean;                            \
            },                                                           \
            benchmark::StatisticUnit::kPercentage                        \
        )                                                                \
        ->Unit(benchmark::kMillisecond);                                 \
                                                                         \
    BENCHMARK(bench_function<ABI, TYPE, Intrinsics::Custom>)             \
        ->Name(std::string("custom ") + simd_name<ABI, TYPE>)            \
        ->Iterations(32)                                                 \
        ->Repetitions(256)                                               \
        ->ReportAggregatesOnly(true)                                     \
        ->ComputeStatistics(                                             \
            "speedup from scalar",                                       \
            [](const std::vector<double>& v) -> double {                 \
                double accum = std::accumulate(v.begin(), v.end(), 0.0); \
                double mean = accum / v.size();                          \
                return baseline<TYPE> / mean;                            \
            },                                                           \
            benchmark::StatisticUnit::kPercentage                        \
        )                                                                \
        ->ComputeStatistics(                                             \
            "speedup from kokkos",                                       \
            [](const std::vector<double>& v) -> double {                 \
                double accum = std::accumulate(v.begin(), v.end(), 0.0); \
                double mean = accum / v.size();                          \
                return baseline_kokkos<TYPE, ABI> / mean;                \
            },                                                           \
            benchmark::StatisticUnit::kPercentage                        \
        )                                                                \
        ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<
              Kokkos::Experimental::simd_abi::scalar,
              float,
              Intrinsics::Kokkos>)
    ->Name(std::string("kokkos ") + simd_name<Kokkos::Experimental::simd_abi::scalar, float>)
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

BENCHMARK(bench_function<
              Kokkos::Experimental::simd_abi::scalar,
              double,
              Intrinsics::Kokkos>)
    ->Name(std::string("kokkos ") + simd_name<Kokkos::Experimental::simd_abi::scalar, double>)
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
GENERATE_BENCHMARK(float, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>)
GENERATE_BENCHMARK(float, Kokkos::Experimental::simd_abi::avx2_fixed_size<8>)
GENERATE_BENCHMARK(double, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>)
#endif

#ifdef KOKKOS_ARCH_AVX512XEON
GENERATE_BENCHMARK(float, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>)
GENERATE_BENCHMARK(float, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>)
GENERATE_BENCHMARK(double, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>)
#endif

BENCHMARK_MAIN();
