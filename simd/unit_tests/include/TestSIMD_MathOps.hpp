#ifndef CEXA_EXPERIMENTAL_TEST_SIMD_MATH_OPS_HPP
#define CEXA_EXPERIMENTAL_TEST_SIMD_MATH_OPS_HPP

#include <SIMDTesting_Utilities.hpp>
#include <SIMDTesting_Ops.hpp>

template<class Abi, class Loader, class BinaryOp, class T>
void host_check_math_op_one_loader(
    BinaryOp binary_op,
    std::size_t n,
    T const* first_args,
    T const* second_args
) {
    Loader loader;
    using simd_type = Kokkos::Experimental::basic_simd<T, Abi>;
    constexpr std::size_t width = simd_type::size();
    for (std::size_t i = 0; i < n; i += width) {
        std::size_t const nremaining = n - i;
        std::size_t const nlanes = Kokkos::min(nremaining, width);
        simd_type first_arg;
        bool const loaded_first_arg =
            loader.host_load(first_args + i, nlanes, first_arg);
        simd_type second_arg;
        bool const loaded_second_arg =
            loader.host_load(second_args + i, nlanes, second_arg);
        if (!(loaded_first_arg && loaded_second_arg))
            continue;

        T expected_val[width];
        for (std::size_t lane = 0; lane < width; ++lane) {
            expected_val[lane] =
                binary_op.on_host(T(first_arg[lane]), T(second_arg[lane]));
        }

        simd_type expected_result;
        expected_result.copy_from(expected_val, Kokkos::Experimental::simd_flag_default);

        simd_type const computed_result = binary_op.on_host(first_arg, second_arg);
        host_check_equality(expected_result, computed_result, nlanes);
    }
}

template<class Abi, class Loader, class UnaryOp, class T>
void host_check_math_op_one_loader(UnaryOp unary_op, std::size_t n, T const* args) {
    Loader loader;
    using simd_type = Kokkos::Experimental::basic_simd<T, Abi>;
    constexpr std::size_t width = simd_type::size();
    for (std::size_t i = 0; i < n; i += width) {
        std::size_t const nremaining = n - i;
        std::size_t const nlanes = Kokkos::min(nremaining, width);
        simd_type arg;
        bool const loaded_arg = loader.host_load(args + i, nlanes, arg);
        if (!loaded_arg)
            continue;

        // if constexpr (std::is_same_v<UnaryOp, cbrt_op> ||
        //               std::is_same_v<UnaryOp, exp_op> ||
        //               std::is_same_v<UnaryOp, log_op>)
        //   arg = Kokkos::abs(arg);
        //
        typename decltype(unary_op.on_host(arg))::value_type expected_val[width];
        for (std::size_t lane = 0; lane < width; ++lane) {
            expected_val[lane] = unary_op.on_host_serial(T(arg[lane]));
        }

        decltype(unary_op.on_host(arg)) expected_result;
        expected_result.copy_from(expected_val, Kokkos::Experimental::simd_flag_default);

        auto computed_result = unary_op.on_host(arg);
        host_check_equality(expected_result, computed_result, nlanes);
    }
}

template<class Abi, class Op, class... T>
inline void host_check_math_op_all_loaders(Op op, std::size_t n, T const*... args) {
    host_check_math_op_one_loader<Abi, load_element_aligned>(op, n, args...);
    host_check_math_op_one_loader<Abi, load_masked>(op, n, args...);
    host_check_math_op_one_loader<Abi, load_as_scalars>(op, n, args...);
    host_check_math_op_one_loader<Abi, load_vector_aligned>(op, n, args...);
}

template<typename Abi, typename DataType, size_t n>
inline void host_check_all_math_ops(
    const DataType (&first_args)[n],
    const DataType (&second_args)[n]
) {
    host_check_math_op_all_loaders<Abi>(exp_op(), n, first_args);
}

template <typename Abi, typename DataType>
inline void host_check_abi_size() {
  using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;
  using mask_type = typename simd_type::mask_type;
  static_assert(simd_type::size() == mask_type::size());
}

template <typename Abi, typename DataType>
inline void host_check_math_ops() {
  if constexpr (is_type_v<Kokkos::Experimental::basic_simd<DataType, Abi>>) {
    constexpr size_t alignment =
        Kokkos::Experimental::basic_simd<DataType, Abi>::size() *
        sizeof(DataType);

    host_check_abi_size<Abi, DataType>();

    if constexpr (!std::is_integral_v<DataType>) {
      alignas(alignment) DataType const first_args[] = {
          0.1, 0.4, 0.5,  0.7, 1.0, 1.5,  -2.0, 10.0,
          0.0, 1.2, -2.8, 3.0, 4.0, -0.1, 5.0,  -0.2};
      alignas(alignment) DataType const second_args[] = {
          1.0,  0.2,  1.1,  1.8, -0.1,  -3.0, -2.4, 1.0,
          13.0, -3.2, -2.1, 3.0, -15.0, -0.5, -0.2, -0.2};
      host_check_all_math_ops<Abi>(first_args, second_args);
    } else {
      if constexpr (std::is_signed_v<DataType>) {
        alignas(alignment) DataType const first_args[] = {
            1, 2, -1, 10, 0, 1, -2, 10, 0, 1, -2, -3, 7, 4, -9, -15};
        alignas(alignment) DataType const second_args[] = {
            1, 2, 1, 1, 1, -3, -2, 1, 13, -3, -2, 10, -15, 7, 2, -10};
        host_check_all_math_ops<Abi>(first_args, second_args);
      } else {
        alignas(alignment) DataType const first_args[] = {
            1, 2, 1, 10, 0, 1, 2, 10, 0, 1, 2, 11, 5, 8, 2, 14};
        alignas(alignment) DataType const second_args[] = {
            1, 2, 1, 1, 1, 3, 2, 1, 13, 3, 2, 3, 6, 20, 5, 14};
        host_check_all_math_ops<Abi>(first_args, second_args);
      }
    }
  }
}

template <typename Abi, typename... DataTypes>
inline void host_check_math_ops_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (host_check_math_ops<Abi, DataTypes>(), ...);
}

template <typename... Abis>
inline void host_check_math_ops_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (host_check_math_ops_all_types<Abis>(DataTypes()), ...);
}

template<class Abi, class DataType, class Op>
inline void host_check_math_op_special_values_all_loaders(Op op) {
    op.template check_special_values<Abi, load_element_aligned, DataType>();
    op.template check_special_values<Abi, load_masked, DataType>();
    op.template check_special_values<Abi, load_as_scalars, DataType>();
    op.template check_special_values<Abi, load_vector_aligned, DataType>();
}

template<typename Abi, typename DataType>
inline void host_check_all_math_ops_special_values(
) {
    host_check_math_op_special_values_all_loaders<Abi, DataType>(exp_op());
}
template <typename Abi, typename DataType>
inline void host_check_math_ops_special_values() {
  if constexpr (is_type_v<Kokkos::Experimental::basic_simd<DataType, Abi>>) {
    constexpr size_t alignment =
        Kokkos::Experimental::basic_simd<DataType, Abi>::size() *
        sizeof(DataType);

    host_check_abi_size<Abi, DataType>();
    host_check_all_math_ops_special_values<Abi, DataType>();
  }
}

template <typename Abi, typename... DataTypes>
inline void host_check_math_ops_special_values_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (host_check_math_ops_special_values<Abi, DataTypes>(), ...);
}

template <typename... Abis>
inline void host_check_math_ops_special_values_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (host_check_math_ops_special_values_all_types<Abis>(DataTypes()), ...);
}


TEST(simd, host_math_ops) {
  host_check_math_ops_all_abis(Kokkos::Experimental::Impl::host_abi_set());
}

TEST(simd, host_math_ops_special_values) {
  host_check_math_ops_special_values_all_abis(Kokkos::Experimental::Impl::host_abi_set());
}

#endif
