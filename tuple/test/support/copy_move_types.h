//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_STD_UTILITIES_TUPLE_CNSTR_TYPES_H
#define LIBCXX_TEST_STD_UTILITIES_TUPLE_CNSTR_TYPES_H

#include "test_allocator.h"
#include <type_traits>

#include <tuple.hpp>
#include <Kokkos_Macros.hpp>
// Types that can be used to test copy/move operations

struct MutableCopy {
  int val{};
  bool alloc_constructed{false};

  KOKKOS_DEFAULTED_FUNCTION constexpr MutableCopy() = default;
  KOKKOS_INLINE_FUNCTION constexpr MutableCopy(int _val) : val(_val) {}
  KOKKOS_DEFAULTED_FUNCTION constexpr MutableCopy(MutableCopy&) = default;
  constexpr MutableCopy(const MutableCopy&)                     = delete;

  // constexpr MutableCopy(std::allocator_arg_t, const test_allocator<int>&,
  //                       MutableCopy& o)
  //     : val(o.val), alloc_constructed(true) {}
};

template <>
struct std::uses_allocator<MutableCopy, test_allocator<int>> : std::true_type {
};

struct ConstCopy {
  int val{};
  bool alloc_constructed{false};

  KOKKOS_DEFAULTED_FUNCTION constexpr ConstCopy() = default;
  KOKKOS_INLINE_FUNCTION constexpr ConstCopy(int _val) : val(_val) {}
  KOKKOS_DEFAULTED_FUNCTION constexpr ConstCopy(const ConstCopy&) = default;
  constexpr ConstCopy(ConstCopy&)                                 = delete;

  // constexpr ConstCopy(std::allocator_arg_t, const test_allocator<int>&,
  //                     const ConstCopy& o)
  //     : val(o.val), alloc_constructed(true) {}
};

template <>
struct std::uses_allocator<ConstCopy, test_allocator<int>> : std::true_type {};

struct MutableMove {
  int val{};
  bool alloc_constructed{false};

  KOKKOS_DEFAULTED_FUNCTION constexpr MutableMove() = default;
  KOKKOS_INLINE_FUNCTION constexpr MutableMove(int _val) : val(_val) {}
  KOKKOS_DEFAULTED_FUNCTION constexpr MutableMove(MutableMove&&) = default;
  constexpr MutableMove(const MutableMove&&)                     = delete;

  // constexpr MutableMove(std::allocator_arg_t, const test_allocator<int>&,
  //                       MutableMove&& o)
  //     : val(o.val), alloc_constructed(true) {}
};

template <>
struct std::uses_allocator<MutableMove, test_allocator<int>> : std::true_type {
};

struct ConstMove {
  int val{};
  bool alloc_constructed{false};

  KOKKOS_DEFAULTED_FUNCTION constexpr ConstMove() = default;
  KOKKOS_INLINE_FUNCTION constexpr ConstMove(int _val) : val(_val) {}
  KOKKOS_INLINE_FUNCTION constexpr ConstMove(const ConstMove&& o)
      : val(o.val) {}
  constexpr ConstMove(ConstMove&&) = delete;

  // constexpr ConstMove(std::allocator_arg_t, const test_allocator<int>&,
  //                     const ConstMove&& o)
  //     : val(o.val), alloc_constructed(true) {}
};

template <>
struct std::uses_allocator<ConstMove, test_allocator<int>> : std::true_type {};

template <class T>
struct ConvertibleFrom {
  T v{};
  bool alloc_constructed{false};

  KOKKOS_DEFAULTED_FUNCTION constexpr ConvertibleFrom() = default;
  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_constructible_v<T, T&>>>
  KOKKOS_INLINE_FUNCTION constexpr ConvertibleFrom(T& _v) : v(_v) {}

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_constructible_v<T, const T&> &&
                                           !std::is_const_v<T>>>
  KOKKOS_INLINE_FUNCTION constexpr ConvertibleFrom(const T& _v) : v(_v) {}

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_constructible_v<T, T&&>>>
  KOKKOS_INLINE_FUNCTION constexpr ConvertibleFrom(T&& _v) : v(std::move(_v)) {}

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_constructible_v<T, const T&&> &&
                                           !std::is_const_v<T>>>
  KOKKOS_INLINE_FUNCTION constexpr ConvertibleFrom(const T&& _v)
      : v(std::move(_v)) {}

  // template <class U, class = std::enable_if_t<
  //                        std::is_constructible_v<ConvertibleFrom, U&&>>>
  // constexpr ConvertibleFrom(std::allocator_arg_t, const test_allocator<int>&,
  //                           U&& _u)
  //     : ConvertibleFrom{std::forward<U>(_u)} {
  //   alloc_constructed = true;
  // }
};

template <class T>
struct std::uses_allocator<ConvertibleFrom<T>, test_allocator<int>>
    : std::true_type {};

template <class T>
struct ExplicitConstructibleFrom {
  T v{};
  bool alloc_constructed{false};

  KOKKOS_DEFAULTED_FUNCTION constexpr explicit ExplicitConstructibleFrom() =
      default;

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_constructible_v<T, T&>>>
  KOKKOS_INLINE_FUNCTION constexpr explicit ExplicitConstructibleFrom(T& _v)
      : v(_v) {}

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_constructible_v<T, const T&> &&
                                           !std::is_const_v<T>>>
  KOKKOS_INLINE_FUNCTION constexpr explicit ExplicitConstructibleFrom(
      const T& _v)
      : v(_v) {}

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_constructible_v<T, T&&>>>
  KOKKOS_INLINE_FUNCTION constexpr explicit ExplicitConstructibleFrom(T&& _v)
      : v(std::move(_v)) {}

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_constructible_v<T, const T&&> &&
                                           !std::is_const_v<T>>>
  KOKKOS_INLINE_FUNCTION constexpr explicit ExplicitConstructibleFrom(
      const T&& _v)
      : v(std::move(_v)) {}

  // template <class U, class = std::enable_if_t<std::is_constructible_v<
  //                        ExplicitConstructibleFrom, U&&>>>
  // constexpr ExplicitConstructibleFrom(std::allocator_arg_t,
  //                                     const test_allocator<int>&, U&& _u)
  //     : ExplicitConstructibleFrom{std::forward<U>(_u)} {
  //   alloc_constructed = true;
  // }
};

template <class T>
struct std::uses_allocator<ExplicitConstructibleFrom<T>, test_allocator<int>>
    : std::true_type {};

struct TracedCopyMove {
  int nonConstCopy       = 0;
  int constCopy          = 0;
  int nonConstMove       = 0;
  int constMove          = 0;
  bool alloc_constructed = false;

  KOKKOS_DEFAULTED_FUNCTION constexpr TracedCopyMove() = default;
  KOKKOS_INLINE_FUNCTION constexpr TracedCopyMove(const TracedCopyMove& other)
      : nonConstCopy(other.nonConstCopy),
        constCopy(other.constCopy + 1),
        nonConstMove(other.nonConstMove),
        constMove(other.constMove) {}
  KOKKOS_INLINE_FUNCTION constexpr TracedCopyMove(TracedCopyMove& other)
      : nonConstCopy(other.nonConstCopy + 1),
        constCopy(other.constCopy),
        nonConstMove(other.nonConstMove),
        constMove(other.constMove) {}

  KOKKOS_INLINE_FUNCTION constexpr TracedCopyMove(TracedCopyMove&& other)
      : nonConstCopy(other.nonConstCopy),
        constCopy(other.constCopy),
        nonConstMove(other.nonConstMove + 1),
        constMove(other.constMove) {}

  KOKKOS_INLINE_FUNCTION constexpr TracedCopyMove(const TracedCopyMove&& other)
      : nonConstCopy(other.nonConstCopy),
        constCopy(other.constCopy),
        nonConstMove(other.nonConstMove),
        constMove(other.constMove + 1) {}

  // template <class U>
  //   requires std::is_constructible_v<TracedCopyMove, U&&>
  // template <class U, class = std::enable_if_t<std::is_constructible_v<TracedCopyMove, U&&>>>
  // constexpr TracedCopyMove(std::allocator_arg_t, const test_allocator<int>&,
  //                          U&& _u)
  //     : TracedCopyMove{std::forward<U>(_u)} {
  //   alloc_constructed = true;
  // }
};

template <>
struct std::uses_allocator<TracedCopyMove, test_allocator<int>>
    : std::true_type {};

// If the constructor tuple(tuple<UTypes...>&) is not available,
// the fallback call to `tuple(const tuple&) = default;` or any other
// constructor that takes const ref would increment the constCopy.
KOKKOS_INLINE_FUNCTION constexpr bool nonConstCopyCtrCalled(
    const TracedCopyMove& obj) {
  return obj.nonConstCopy == 1 && obj.constCopy == 0 && obj.constMove == 0 &&
         obj.nonConstMove == 0;
}

// If the constructor tuple(const tuple<UTypes...>&&) is not available,
// the fallback call to `tuple(const tuple&) = default;` or any other
// constructor that takes const ref would increment the constCopy.
KOKKOS_INLINE_FUNCTION constexpr bool constMoveCtrCalled(
    const TracedCopyMove& obj) {
  return obj.nonConstMove == 0 && obj.constMove == 1 && obj.constCopy == 0 &&
         obj.nonConstCopy == 0;
}

struct NoConstructorFromInt {};

struct CvtFromTupleRef : TracedCopyMove {
  KOKKOS_DEFAULTED_FUNCTION constexpr CvtFromTupleRef() = default;
  KOKKOS_INLINE_FUNCTION constexpr CvtFromTupleRef(
      cexa::tuple<CvtFromTupleRef>& other)
      : TracedCopyMove(static_cast<TracedCopyMove&>(cexa::get<0>(other))) {}
};

struct ExplicitCtrFromTupleRef : TracedCopyMove {
  KOKKOS_DEFAULTED_FUNCTION constexpr explicit ExplicitCtrFromTupleRef() =
      default;
  KOKKOS_INLINE_FUNCTION constexpr explicit ExplicitCtrFromTupleRef(
      cexa::tuple<ExplicitCtrFromTupleRef>& other)
      : TracedCopyMove(static_cast<TracedCopyMove&>(cexa::get<0>(other))) {}
};

struct CvtFromConstTupleRefRef : TracedCopyMove {
  KOKKOS_DEFAULTED_FUNCTION constexpr CvtFromConstTupleRefRef() = default;
  KOKKOS_INLINE_FUNCTION constexpr CvtFromConstTupleRefRef(
      const cexa::tuple<CvtFromConstTupleRefRef>&& other)
      : TracedCopyMove(
            static_cast<const TracedCopyMove&&>(cexa::get<0>(other))) {}
};

struct ExplicitCtrFromConstTupleRefRef : TracedCopyMove {
  KOKKOS_DEFAULTED_FUNCTION constexpr explicit ExplicitCtrFromConstTupleRefRef() =
      default;
  KOKKOS_INLINE_FUNCTION constexpr explicit ExplicitCtrFromConstTupleRefRef(
      cexa::tuple<const ExplicitCtrFromConstTupleRefRef>&& other)
      : TracedCopyMove(
            static_cast<const TracedCopyMove&&>(cexa::get<0>(other))) {}
};

template <class T>
KOKKOS_INLINE_FUNCTION void conversion_test(T);

// TODO: convert to type trait
// NOTE: This is only used in alloc tests
// template <class T, class... Args>
// concept ImplicitlyConstructible = requires(Args&&... args) {
//   conversion_test<T>({std::forward<Args>(args)...});
// };

struct CopyAssign {
  int val{};

  KOKKOS_DEFAULTED_FUNCTION constexpr CopyAssign() = default;
  KOKKOS_INLINE_FUNCTION constexpr CopyAssign(int v) : val(v) {}

  KOKKOS_DEFAULTED_FUNCTION constexpr CopyAssign& operator=(const CopyAssign&) =
      default;

  constexpr const CopyAssign& operator=(const CopyAssign&) const = delete;
  constexpr CopyAssign& operator=(CopyAssign&&)                  = delete;
  constexpr const CopyAssign& operator=(CopyAssign&&) const      = delete;
};

struct ConstCopyAssign {
  mutable int val{};

  KOKKOS_DEFAULTED_FUNCTION constexpr ConstCopyAssign() = default;
  KOKKOS_INLINE_FUNCTION constexpr ConstCopyAssign(int v) : val(v) {}

  KOKKOS_INLINE_FUNCTION constexpr const ConstCopyAssign& operator=(
      const ConstCopyAssign& other) const {
    val = other.val;
    return *this;
  }

  constexpr ConstCopyAssign& operator=(const ConstCopyAssign&)        = delete;
  constexpr ConstCopyAssign& operator=(ConstCopyAssign&&)             = delete;
  constexpr const ConstCopyAssign& operator=(ConstCopyAssign&&) const = delete;
};

struct MoveAssign {
  int val{};

  KOKKOS_DEFAULTED_FUNCTION constexpr MoveAssign() = default;
  KOKKOS_INLINE_FUNCTION constexpr MoveAssign(int v) : val(v) {}

  KOKKOS_DEFAULTED_FUNCTION constexpr MoveAssign& operator=(MoveAssign&&) =
      default;

  constexpr MoveAssign& operator=(const MoveAssign&)             = delete;
  constexpr const MoveAssign& operator=(const MoveAssign&) const = delete;
  constexpr const MoveAssign& operator=(MoveAssign&&) const      = delete;
};

struct ConstMoveAssign {
  mutable int val{};

  KOKKOS_DEFAULTED_FUNCTION constexpr ConstMoveAssign() = default;
  KOKKOS_INLINE_FUNCTION constexpr ConstMoveAssign(int v) : val(v) {}

  KOKKOS_INLINE_FUNCTION constexpr const ConstMoveAssign& operator=(
      ConstMoveAssign&& other) const {
    val = other.val;
    return *this;
  }

  constexpr ConstMoveAssign& operator=(const ConstMoveAssign&) = delete;
  constexpr const ConstMoveAssign& operator=(const ConstMoveAssign&) const =
      delete;
  constexpr ConstMoveAssign& operator=(ConstMoveAssign&&) = delete;
};

template <class T>
struct AssignableFrom {
  T v{};

  KOKKOS_DEFAULTED_FUNCTION constexpr AssignableFrom() = default;

  template <class U, class = std::enable_if_t<std::is_constructible_v<T, U&&>>>
  KOKKOS_INLINE_FUNCTION constexpr AssignableFrom(U&& u)
      : v(std::forward<U>(u)) {}

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_copy_assignable_v<T>>>
  KOKKOS_INLINE_FUNCTION constexpr AssignableFrom& operator=(const T& t) {
    v = t;
    return *this;
  }

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_move_assignable_v<T>>>
  KOKKOS_INLINE_FUNCTION constexpr AssignableFrom& operator=(T&& t) {
    v = std::move(t);
    return *this;
  }

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_assignable_v<const T&, const T&>>>
  KOKKOS_INLINE_FUNCTION constexpr const AssignableFrom& operator=(const T& t) const {
    v = t;
    return *this;
  }

  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_assignable_v<const T&, T&&>>>
  KOKKOS_INLINE_FUNCTION constexpr const AssignableFrom& operator=(T&& t) const {
    v = std::move(t);
    return *this;
  }
};

struct TracedAssignment {
  int copyAssign              = 0;
  mutable int constCopyAssign = 0;
  int moveAssign              = 0;
  mutable int constMoveAssign = 0;

  KOKKOS_DEFAULTED_FUNCTION constexpr TracedAssignment() = default;

  KOKKOS_INLINE_FUNCTION constexpr TracedAssignment& operator=(const TracedAssignment&) {
    copyAssign++;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const TracedAssignment& operator=(const TracedAssignment&) const {
    constCopyAssign++;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr TracedAssignment& operator=(TracedAssignment&&) {
    moveAssign++;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr const TracedAssignment& operator=(TracedAssignment&&) const {
    constMoveAssign++;
    return *this;
  }
};
#endif
