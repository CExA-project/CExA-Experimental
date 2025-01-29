#ifndef KOKKOS_VARIANT_HPP
#define KOKKOS_VARIANT_HPP

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>

#if KOKKOS_COMPILER_NVCC >= 1240
#include <cuda/std/variant>
#define KOKKOS_VARIANT_PREFIX cuda::std
#else
#include "./mpark_variant/include/mpark/variant.hpp"
#define KOKKOS_VARIANT_PREFIX mpark
#endif
#elif defined (KOKKOS_ENABLE_HIP) || \
  defined (KOKKOS_ENABLE_SYCL)
#include "./mpark_variant/include/mpark/variant.hpp"
#define KOKKOS_VARIANT_PREFIX mpark
#else
#include <variant>
#define KOKKOS_VARIANT_PREFIX std
#endif

// Create an alias of function F called G
#define KOKKOS_ALIAS_FUNCTION(F, G)                  \
  template <typename... Args>                        \
  KOKKOS_FORCEINLINE_FUNCTION auto G(Args&&... args) \
    -> decltype(F(std::forward<Args>(args)...)) {    \
    return F(std::forward<Args>(args)...);           \
  }

namespace Cexa::Experimental {
  template <class... types>
  using variant = KOKKOS_VARIANT_PREFIX::variant<types...>;
  KOKKOS_ALIAS_FUNCTION(KOKKOS_VARIANT_PREFIX::visit, visit);
  KOKKOS_ALIAS_FUNCTION(KOKKOS_VARIANT_PREFIX::holds_alternative, holds_alternative);
  KOKKOS_ALIAS_FUNCTION(KOKKOS_VARIANT_PREFIX::get, get);
}

#undef KOKKOS_VARIANT_PREFIX
#undef KOKKOS_ALIAS_FUNCTION
#endif
