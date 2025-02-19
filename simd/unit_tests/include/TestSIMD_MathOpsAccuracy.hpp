// SPDX-FileCopyrightText: 2025 CExA-project
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef CEXA_EXPERIMENTAL_TEST_SIMD_MATH_OPS_ACCURACY_HPP
#define CEXA_EXPERIMENTAL_TEST_SIMD_MATH_OPS_ACCURACY_HPP

#include <SIMDTesting_Ops.hpp>

template<class Abi, class DataType, class UnaryOp>
void host_check_math_op_accuracy(UnaryOp op) {
    using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;

    constexpr std::size_t width = simd_type::size();

    std::uint32_t values[width];
    for (std::uint32_t i = 0; i < width; i++) {
        values[i] = i;
    }

    simd_type vec;
    typename decltype(op.on_host(vec))::value_type expected[width];
    decltype(op.on_host(vec)) expected_result;

    for (std::size_t i = 0;
         i <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
         i += width) {
        vec.copy_from(
            reinterpret_cast<DataType*>(values),
            Kokkos::Experimental::simd_flag_default
        );

        for (std::size_t lane = 0; lane < width; ++lane) {
            expected[lane] = op.on_host_serial(DataType(vec[lane]));
        }

        expected_result.copy_from(expected, Kokkos::Experimental::simd_flag_default);
        auto computed_result = op.on_host(vec);

        host_check_equality(expected_result, computed_result, width);

        for (auto& value: values) {
            value += width;
        }
    }
}

template<class Abi, class DataType>
inline void host_check_all_math_ops_accuracy() {
    if constexpr (!std::is_same_v<Abi, Kokkos::Experimental::simd_abi::scalar>
                  && is_type_v<Kokkos::Experimental::basic_simd<DataType, Abi>>) {
        host_check_math_op_accuracy<Abi, DataType>(exp_op());
    }
}

template<typename Abi, typename... DataTypes>
inline void
host_check_math_ops_accuracy_all_types(Kokkos::Experimental::Impl::data_types<
                                       DataTypes...>) {
    (host_check_all_math_ops_accuracy<Abi, DataTypes>(), ...);
}

template<typename... Abis>
inline void
host_check_math_ops_accuracy_all_abis(Kokkos::Experimental::Impl::abi_set<Abis...>) {
    // using DataTypes = Kokkos::Experimental::Impl::data_type_set;
    using DataTypes = Kokkos::Experimental::Impl::data_types<float>;
    (host_check_math_ops_accuracy_all_types<Abis>(DataTypes()), ...);
}

TEST(simd, host_math_ops_accuracy) {
  host_check_math_ops_accuracy_all_abis(Kokkos::Experimental::Impl::host_abi_set());
}

#endif
