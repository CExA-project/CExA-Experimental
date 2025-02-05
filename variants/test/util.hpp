// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Core.hpp>
#include <mpark/config.hpp>

enum Qual { Ptr, ConstPtr, LRef, ConstLRef, RRef, ConstRRef };

struct get_qual_t {
  KOKKOS_FUNCTION constexpr Qual operator()(int *) const { return Ptr; }
  KOKKOS_FUNCTION constexpr Qual operator()(const int *) const {
    return ConstPtr;
  }
  KOKKOS_FUNCTION constexpr Qual operator()(int &) const { return LRef; }
  KOKKOS_FUNCTION constexpr Qual operator()(const int &) const {
    return ConstLRef;
  }
  KOKKOS_FUNCTION constexpr Qual operator()(int &&) const { return RRef; }
  KOKKOS_FUNCTION constexpr Qual operator()(const int &&) const {
    return ConstRRef;
  }
};

constexpr get_qual_t get_qual{};

#ifdef MPARK_EXCEPTIONS
struct CopyConstruction : std::exception {};
struct CopyAssignment : std::exception {};
struct MoveConstruction : std::exception {};
struct MoveAssignment : std::exception {};

struct copy_thrower_t {
  KOKKOS_FUNCTION constexpr copy_thrower_t() {}
  KOKKOS_FUNCTION [[noreturn]] copy_thrower_t(const copy_thrower_t &) {
    throw CopyConstruction{};
  }
  KOKKOS_FUNCTION copy_thrower_t(copy_thrower_t &&) = default;
  KOKKOS_FUNCTION copy_thrower_t &operator=(const copy_thrower_t &) {
    throw CopyAssignment{};
  }
  KOKKOS_FUNCTION copy_thrower_t &operator=(copy_thrower_t &&) = default;
};

KOKKOS_INLINE_FUNCTION bool operator<(const copy_thrower_t &,
                                      const copy_thrower_t &) noexcept {
  return false;
}

KOKKOS_INLINE_FUNCTION bool operator>(const copy_thrower_t &,
                                      const copy_thrower_t &) noexcept {
  return false;
}

KOKKOS_INLINE_FUNCTION bool operator<=(const copy_thrower_t &,
                                       const copy_thrower_t &) noexcept {
  return true;
}

KOKKOS_INLINE_FUNCTION bool operator>=(const copy_thrower_t &,
                                       const copy_thrower_t &) noexcept {
  return true;
}

KOKKOS_INLINE_FUNCTION bool operator==(const copy_thrower_t &,
                                       const copy_thrower_t &) noexcept {
  return true;
}

KOKKOS_INLINE_FUNCTION bool operator!=(const copy_thrower_t &,
                                       const copy_thrower_t &) noexcept {
  return false;
}

struct move_thrower_t {
  KOKKOS_FUNCTION constexpr move_thrower_t() {}
  KOKKOS_FUNCTION move_thrower_t(const move_thrower_t &) = default;
  KOKKOS_FUNCTION [[noreturn]] move_thrower_t(move_thrower_t &&) {
    throw MoveConstruction{};
  }
  KOKKOS_FUNCTION move_thrower_t &operator=(const move_thrower_t &) = default;
  KOKKOS_FUNCTION move_thrower_t &operator=(move_thrower_t &&) {
    throw MoveAssignment{};
  }
};

KOKKOS_INLINE_FUNCTION bool operator<(const move_thrower_t &,
                                      const move_thrower_t &) noexcept {
  return false;
}

KOKKOS_INLINE_FUNCTION bool operator>(const move_thrower_t &,
                                      const move_thrower_t &) noexcept {
  return false;
}

KOKKOS_INLINE_FUNCTION bool operator<=(const move_thrower_t &,
                                       const move_thrower_t &) noexcept {
  return true;
}

KOKKOS_INLINE_FUNCTION bool operator>=(const move_thrower_t &,
                                       const move_thrower_t &) noexcept {
  return true;
}

KOKKOS_INLINE_FUNCTION bool operator==(const move_thrower_t &,
                                       const move_thrower_t &) noexcept {
  return true;
}

KOKKOS_INLINE_FUNCTION bool operator!=(const move_thrower_t &,
                                       const move_thrower_t &) noexcept {
  return false;
}

#endif
