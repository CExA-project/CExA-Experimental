#ifndef KOKKOS_VARIANT_HPP
#define KOKKOS_VARIANT_HPP

#include <Kokkos_Core.hpp>

// We need a function called "swap" (not kokkos_swap like the one in Kokkos)
namespace cexa::experimental {

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr std::enable_if_t<
    std::is_move_constructible_v<T> && std::is_move_assignable_v<T>>
swap(T &a, T &b) noexcept(std::is_nothrow_move_constructible_v<T>
                              &&std::is_nothrow_move_assignable_v<T>) {
  Kokkos::kokkos_swap(a, b);
}

template <class T, std::size_t N>
KOKKOS_FORCEINLINE_FUNCTION constexpr std::enable_if_t<
    Kokkos::Impl::is_swappable<T>::value>
swap(T (&a)[N], T (&b)[N]) noexcept(Kokkos::Impl::is_nothrow_swappable_v<T>) {
  Kokkos::kokkos_swap(a, b);
}
}  // namespace cexa::experimental

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>

#if KOKKOS_COMPILER_NVCC >= 1240
#include <cuda/std/variant>
#define KOKKOS_VARIANT_PREFIX cuda::std
#else
#include "mpark/variant.hpp"
#define KOKKOS_VARIANT_PREFIX mpark
#endif
#elif defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
#include "mpark/variant.hpp"
#define KOKKOS_VARIANT_PREFIX mpark
#else
#include <variant>
#define KOKKOS_VARIANT_PREFIX std
#endif

namespace cexa::experimental {
template <class... types>
using variant   = KOKKOS_VARIANT_PREFIX::variant<types...>;
using monostate = KOKKOS_VARIANT_PREFIX::monostate;

template <typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto visit(Args &&...args)
    -> decltype(KOKKOS_VARIANT_PREFIX::visit(std::forward<Args>(args)...)) {
  return KOKKOS_VARIANT_PREFIX::visit(std::forward<Args>(args)...);
}

template <typename T, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION constexpr bool holds_alternative(
    const variant<Ts...> &v) noexcept {
  return KOKKOS_VARIANT_PREFIX::holds_alternative<T>(v);
}

// get
template <std::size_t I, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr KOKKOS_VARIANT_PREFIX::
    variant_alternative_t<I, variant<Args...>> &
    get(variant<Args...> &arg) {
  return KOKKOS_VARIANT_PREFIX::get<I>(arg);
}

template <std::size_t I, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr KOKKOS_VARIANT_PREFIX::
    variant_alternative_t<I, variant<Args...>> &&
    get(variant<Args...> &&arg) {
  return KOKKOS_VARIANT_PREFIX::get<I>(std::forward<Args...>(arg));
}

template <std::size_t I, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr const KOKKOS_VARIANT_PREFIX::
    variant_alternative_t<I, variant<Args...>> &
    get(const variant<Args...> &arg) {
  return KOKKOS_VARIANT_PREFIX::get<I>(arg);
}

template <std::size_t I, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr const KOKKOS_VARIANT_PREFIX::
    variant_alternative_t<I, variant<Args...>> &&
    get(const variant<Args...> &&arg) {
  return KOKKOS_VARIANT_PREFIX::get<I>(std::forward<Args...>(arg));
}

template <typename T, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr T &get(variant<Args...> &arg) {
  return KOKKOS_VARIANT_PREFIX::get<T, Args...>(arg);
}

template <typename T, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr T &&get(variant<Args...> &&arg) {
  return KOKKOS_VARIANT_PREFIX::get<T>(std::forward<variant<Args...>>(arg));
}

template <typename T, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr const T &get(
    const variant<Args...> &arg) {
  return KOKKOS_VARIANT_PREFIX::get<T, Args...>(arg);
}

template <typename T, typename... Args>
KOKKOS_FORCEINLINE_FUNCTION constexpr const T &&get(
    const variant<Args...> &&arg) {
  return KOKKOS_VARIANT_PREFIX::get<T>(
      std::forward<const variant<Args...>>(arg));
}

// get_if
template <std::size_t I, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto get_if(variant<Ts...> *v) noexcept {
  return KOKKOS_VARIANT_PREFIX::get_if<I>(v);
}

template <std::size_t I, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto get_if(
    const variant<Ts...> *v) noexcept {
  return KOKKOS_VARIANT_PREFIX::get_if<I>(v);
}

template <typename T, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto get_if(variant<Ts...> *v) noexcept {
  return KOKKOS_VARIANT_PREFIX::get_if<T>(v);
}

template <typename T, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto get_if(
    const variant<Ts...> *v) noexcept {
  return KOKKOS_VARIANT_PREFIX::get_if<T>(v);
}

// in_place...
using in_place_t = KOKKOS_VARIANT_PREFIX::in_place_t;
template <std::size_t I>
using in_place_index_t = KOKKOS_VARIANT_PREFIX::in_place_index_t<I>;
template <typename T>
using in_place_type_t = KOKKOS_VARIANT_PREFIX::in_place_type_t<T>;

// bad_variant_access
using bad_variant_access = KOKKOS_VARIANT_PREFIX::bad_variant_access;
}  // namespace cexa::experimental

#undef KOKKOS_VARIANT_PREFIX
#undef KOKKOS_ALIAS_FUNCTION
#endif
