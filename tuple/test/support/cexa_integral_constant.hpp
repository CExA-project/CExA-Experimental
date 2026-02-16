#pragma once

#include <Kokkos_Macros.hpp>

// device-compatible replacement of std::integral_constant used in the tests

namespace cexa::testing {
template <class T, T v>
struct integral_constant {
  using value_type = T;
  using type       = integral_constant<T, v>;

  constexpr static T value = v;

  KOKKOS_INLINE_FUNCTION constexpr operator T() const noexcept { return v; }
  KOKKOS_INLINE_FUNCTION constexpr T operator()() const noexcept { return v; }
};
}  // namespace cexa::testing
