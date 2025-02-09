#ifndef CEXA_EXPERIMENTAL_SIMD_TESTING_OPS_HPP
#define CEXA_EXPERIMENTAL_SIMD_TESTING_OPS_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
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

    template<bool check_relative_error, class Abi, class Loader, typename T>
    void check_special_values() {
        if constexpr (!std::is_integral_v<T>) {
            using simd_type = Kokkos::Experimental::basic_simd<T, Abi>;
            constexpr std::size_t width = simd_type::size();

            gtest_checker checker;
            simd_type computed;
            T computed_serial;

            // nan
            simd_type nan(std::numeric_limits<T>::quiet_NaN());
            computed = on_host(nan);
            computed_serial = on_host_serial(nan[0]);

            for (std::size_t lane = 0; lane < width; lane++) {
                checker.truth(std::isnan(computed[lane]) && std::isnan(computed_serial));
            }

            T tested_values[] = {
                0.0,
                std::numeric_limits<T>::infinity(),
                -std::numeric_limits<T>::infinity(),
                -103,
                88.7,
                -9.30327e+07,
                -2.38164398e+10,
                2.38164398e+10
            };

            load_masked loader;
            for (std::size_t i = 0; i < std::size(tested_values); i += width) {
                std::size_t nlanes = std::min(width, std::size(tested_values) - i);
                simd_type vec;
                loader.host_load(
                    tested_values + i,
                    nlanes,
                    vec
                );
                computed = on_host(vec);

                for (std::size_t j = 0; j < nlanes; j++) {
                    computed_serial = on_host_serial(tested_values[i + j]);
                    if constexpr (check_relative_error) {
                        checker.closeness(computed_serial, computed[j]);
                    } else {
                        checker.equality(computed_serial, computed[j]);
                    }
                }
            }

        }
    }
};

#endif
