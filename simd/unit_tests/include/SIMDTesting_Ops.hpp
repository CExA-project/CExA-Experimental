#ifndef CEXA_EXPERIMENTAL_SIMD_TESTING_OPS_HPP
#define CEXA_EXPERIMENTAL_SIMD_TESTING_OPS_HPP

#include <cmath>
#include <limits>
#include <type_traits>

#include <Kokkos_SIMD_AVX_Math.hpp>
#include <SIMDTesting_Utilities.hpp>

class exp_op {
public:
    template<typename T>
    auto on_host(T const& a) const {
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
        return Kokkos::Experimental::exp(a);
#else
        return Kokkos::exp(a);
#endif
    }

    template<typename T>
    auto on_host_serial(T const& a) const {
        return Kokkos::exp(a);
    }

    template<class Abi, class Loader, typename T>
    void check_special_values() {
        if constexpr (!std::is_integral_v<T>) {
            using simd_type = Kokkos::Experimental::basic_simd<T, Abi>;
            constexpr std::size_t width = simd_type::size();

            gtest_checker checker;
            simd_type computed;
            T computed_serial;

            // +inf
            simd_type inf(std::numeric_limits<T>::infinity());
            computed = on_host(inf);
            computed_serial = on_host_serial(inf[0]);

            for (std::size_t lane = 0; lane < width; lane++) {
                checker.equality(computed[lane], computed_serial);
            }

            // -inf
            simd_type negative_inf(-std::numeric_limits<T>::infinity());
            computed = on_host(negative_inf);
            computed_serial = on_host_serial(negative_inf[0]);

            for (std::size_t lane = 0; lane < width; lane++) {
                checker.equality(computed[lane], computed_serial);
            }

            // nan
            simd_type nan(std::numeric_limits<T>::quiet_NaN());
            computed = on_host(nan);
            computed_serial = on_host_serial(nan[0]);

            for (std::size_t lane = 0; lane < width; lane++) {
                checker.truth(std::isnan(computed[lane]) && std::isnan(computed_serial));
            }

            T special_values[] = {0.0, -103, 88.7, -9.30327e+07, -2.38164398e+10, 2.38164398e+10};
            for (auto value: special_values) {
                // subnormal
                simd_type vec(value);
                computed = on_host(vec);
                computed_serial = on_host_serial(vec[0]);

                for (std::size_t lane = 0; lane < width; lane++) {
                    checker.equality(computed[lane], computed_serial);
                }
            }
        }
    }
};

#endif
