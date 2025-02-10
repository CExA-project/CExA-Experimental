#include <benchmark/benchmark.h>
#include <cassert>
#include <immintrin.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
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
            Cexa::Experimental::simd::avx2::exp4d(static_cast<__m256d>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<8>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx2_fixed_size<8>> const& x) {
    return Kokkos::Experimental::
        basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<8>>(
            Cexa::Experimental::simd::avx2::exp8f(static_cast<__m256>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx2_fixed_size<4>> const& x) {
    return Kokkos::Experimental::
        basic_simd<float, Kokkos::Experimental::simd_abi::avx2_fixed_size<4>>(
            Cexa::Experimental::simd::avx2::exp4f(static_cast<__m128>(x))
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
            Cexa::Experimental::simd::avx512::exp8d(static_cast<__m512d>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx512_fixed_size<16>> const& x) {
    return Kokkos::Experimental::
        basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>>(
            Cexa::Experimental::simd::avx512::exp16f(static_cast<__m512>(x))
        );
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Kokkos::Experimental::
    basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>
    custom_exp(Kokkos::Experimental::basic_simd<
               float,
               Kokkos::Experimental::simd_abi::avx512_fixed_size<8>> const& x) {
    return Kokkos::Experimental::
        basic_simd<float, Kokkos::Experimental::simd_abi::avx512_fixed_size<8>>(
            Cexa::Experimental::simd::avx512::exp8f(static_cast<__m256>(x))
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
constexpr Kokkos::Experimental::basic_simd<data_type, Abi> KOKKOS_FORCEINLINE_FUNCTION
tested_exp(Kokkos::Experimental::basic_simd<data_type, Abi> const& x) {
    if constexpr (intrinsics == Intrinsics::Kokkos) {
        return Kokkos::exp(x);
    } else {
        return custom_exp(x);
    }
}

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

template<typename Abi, typename data_type, Intrinsics intrinsics>
static void bench_function(benchmark::State& state) {
    using simd_type = Kokkos::Experimental::basic_simd<data_type, Abi>;
    constexpr std::size_t width = simd_type::size();

    std::size_t samples = 10000000;

    data_type* data_test =
        new (std::align_val_t(width * sizeof(data_type))) data_type[samples];
    setup<data_type>(data_test, samples);
    data_type* result =
        new (std::align_val_t(width * sizeof(data_type))) data_type[samples];

    for (auto _: state) {
        Kokkos::parallel_for(
            "loop",
            Kokkos::RangePolicy(0, samples / width),
            KOKKOS_LAMBDA(const std::size_t i) {
                simd_type vec;
                vec.copy_from(
                    &data_test[i * width],
                    Kokkos::Experimental::simd_flag_aligned
                );
                simd_type res;
                if constexpr (intrinsics == Intrinsics::Kokkos) {
                    benchmark::DoNotOptimize(res = Kokkos::exp(vec));
                } else {
                    benchmark::DoNotOptimize(res = custom_exp(vec));
                }
                res.copy_to(&result[i * width], Kokkos::Experimental::simd_flag_aligned);
            }
        );
    }

    benchmark::DoNotOptimize(result[state.bytes_processed() % samples]);

    delete[] data_test;
    delete[] result;
}

#define GENERATE_BENCHMARK(TYPE, ABI)                         \
    BENCHMARK(bench_function<ABI, TYPE, Intrinsics::Kokkos>)  \
        ->Name(std::string("kokkos ") + simd_name<ABI, TYPE>) \
        ->ReportAggregatesOnly(true)                          \
        ->Unit(benchmark::kMillisecond);                      \
                                                              \
    BENCHMARK(bench_function<ABI, TYPE, Intrinsics::Custom>)  \
        ->Name(std::string("custom ") + simd_name<ABI, TYPE>) \
        ->ReportAggregatesOnly(true)                          \
        ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<
              Kokkos::Experimental::simd_abi::scalar,
              float,
              Intrinsics::Kokkos>)
    ->Name(std::string("kokkos ") + simd_name<Kokkos::Experimental::simd_abi::scalar, float>)
    ->ReportAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bench_function<
              Kokkos::Experimental::simd_abi::scalar,
              double,
              Intrinsics::Kokkos>)
    ->Name(std::string("kokkos ") + simd_name<Kokkos::Experimental::simd_abi::scalar, double>)
    ->ReportAggregatesOnly(true)
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

int main(int argc, char** argv) {
    char arg0_default[] = "benchmark";
    char* args_default = arg0_default;
    if (!argv) {
        argc = 1;
        argv = &args_default;
    }
    Kokkos::ScopeGuard guard(argc, argv);

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
