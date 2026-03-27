// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#pragma once

#include <Kokkos_Macros.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
#include <compare>
#endif

#include "macros.hpp"
#include "traits.hpp"
#include "helper.hpp"
#include "tuple_fwd.hpp"

namespace cexa {

namespace impl {

template <std::size_t I, typename... Ts>
struct nth_type;
template <typename T, typename... Ts>
struct nth_type<0, T, Ts...> {
  using type = T;
};
template <std::size_t I, typename T, typename... Ts>
struct nth_type<I, T, Ts...> : nth_type<I - 1, Ts...> {};

#define FWD(x) std::forward<decltype(x)>(x)
// #define FWD(x) static_cast<decltype(x)>(x)

template <class Tuple, class UTuple>
struct all_types_constructible : std::false_type {};

template <class... Types, class UTuple>
struct all_types_constructible<tuple<Types...>, UTuple> {
  using t = void;
  template <class Seq>
  struct all_types_constructible_helper;
  template <std::size_t... Ints>
  struct all_types_constructible_helper<std::index_sequence<Ints...>> {
    static constexpr bool value =
        (std::is_constructible_v<Types,
                                 decltype(get<Ints>(std::declval<UTuple>()))> &&
         ...);
  };

  static constexpr bool value = all_types_constructible_helper<
      decltype(std::index_sequence_for<Types...>{})>::value;
};

template <class Tuple, class UTuple>
inline constexpr bool all_types_constructible_v =
    all_types_constructible<Tuple, UTuple>::value;

template <class UTuple, class Tuple>
struct all_types_convertible : std::false_type {};

template <class... Types, class UTuple>
struct all_types_convertible<UTuple, tuple<Types...>> {
  template <class Seq>
  struct all_types_convertible_helper;
  template <std::size_t... Ints>
  struct all_types_convertible_helper<std::index_sequence<Ints...>> {
    static constexpr bool value =
        (std::is_convertible_v<decltype(get<Ints>(FWD(std::declval<UTuple>()))),
                               Types> &&
         ...);
  };

  static constexpr bool value = all_types_convertible_helper<
      decltype(std::index_sequence_for<Types...>{})>::value;
};

template <class UTuple, class Tuple>
inline constexpr bool all_types_convertible_v =
    all_types_convertible<UTuple, Tuple>::value;

template <class T>
struct is_pair : std::false_type {};

template <class T, class U>
struct is_pair<std::pair<T, U>> : std::true_type {};

template <class Tuple, class UTuple>
struct any_types_reference_constructs_from_temporary : std::false_type {};

template <class... Types, class UTuple>
struct any_types_reference_constructs_from_temporary<tuple<Types...>, UTuple> {
  template <class Seq>
  struct any_types_reference_constructs_from_temporary_helper;
  template <std::size_t... Ints>
  struct any_types_reference_constructs_from_temporary_helper<
      std::index_sequence<Ints...>> {
    static constexpr bool value =
        (impl::reference_constructs_from_temporary_v<
             Types, decltype(get<Ints>(FWD(std::declval<UTuple>())))> ||
         ...);
  };

  static constexpr bool value =
      any_types_reference_constructs_from_temporary_helper<
          decltype(std::index_sequence_for<Types...>{})>::value;
};

template <class Tuple, class UTuple>
inline constexpr bool any_types_reference_constructs_from_temporary_v =
    any_types_reference_constructs_from_temporary<Tuple, UTuple>::value;

template <typename... Types>
struct store;

template <class T>
struct is_store : std::false_type {};

template <class... Types>
struct is_store<store<Types...>> : std::true_type {};

template <class T>
inline constexpr bool is_store_v = is_store<std::remove_cvref_t<T>>::value;

template <>
struct store<> {
  KOKKOS_DEFAULTED_FUNCTION constexpr store()             = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr store(const store&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr store(store&&)      = default;
#if defined(CEXA_HAS_CXX23)
  KOKKOS_DEFAULTED_FUNCTION constexpr store(store&) noexcept {}
  KOKKOS_DEFAULTED_FUNCTION constexpr store(const store&&) noexcept {};
#endif
  KOKKOS_DEFAULTED_FUNCTION constexpr ~store() = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr store& operator=(const store&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr store& operator=(store&&)      = default;

#if defined(CEXA_HAS_CXX23)
  KOKKOS_DEFAULTED_FUNCTION constexpr const store& operator=(
      const store&) const noexcept {
    return *this;
  }
  KOKKOS_DEFAULTED_FUNCTION constexpr const store& operator=(
      store&&) const noexcept {
    return *this;
  }
#endif

  KOKKOS_INLINE_FUNCTION constexpr void swap(store&) noexcept {}
#if defined(CEXA_HAS_CXX23)
  KOKKOS_INLINE_FUNCTION constexpr void swap(const store&) const noexcept {}
#endif

#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
  KOKKOS_DEFAULTED_FUNCTION auto operator<=>(const store&) const = default;
#else
  KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(const store<>&,
                                                          const store<>&) {
    return true;
  }
  KOKKOS_INLINE_FUNCTION friend constexpr bool operator<(const store<>&,
                                                         const store<>&) {
    return false;
  }
#endif
};

template <class T, class... Types>
struct store<T, Types...> {
  T value{};
  store<Types...> rest;

  KOKKOS_DEFAULTED_FUNCTION constexpr store() = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr store(const store& other) = default;
  template <class Dummy = void,
            class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                           std::is_move_constructible_v<T>>>
  KOKKOS_INLINE_FUNCTION constexpr explicit store(store&& other)
      : value(FWD(other.value)), rest(FWD(other.rest)) {}

  template <
      typename U, typename... UTypes,
      class = std::enable_if_t<!is_store_v<U> && (!is_store_v<UTypes> && ...)>>
  KOKKOS_INLINE_FUNCTION constexpr store(U&& u, UTypes&&... args)
      : value(FWD(u)), rest(FWD(args)...) {}

  KOKKOS_DEFAULTED_FUNCTION constexpr ~store() = default;

  template <
      class U, class... UTypes,
      class = std::enable_if_t<sizeof...(UTypes) == sizeof...(Types) &&
                               !(std::is_same_v<U, T> &&
                                 (std::is_same_v<UTypes, Types> && ...)) &&
                               std::is_assignable_v<T&, const U&>>>
  KOKKOS_INLINE_FUNCTION constexpr store& operator=(
      const store<U, UTypes...>& other) {
    value = other.value;
    rest  = other.rest;
    return *this;
  }
  template <
      class U, class... UTypes,
      class = std::enable_if_t<sizeof...(UTypes) == sizeof...(Types) &&
                               !(std::is_same_v<U, T> &&
                                 (std::is_same_v<UTypes, Types> && ...)) &&
                               std::is_assignable_v<T&, U&&>>>
  KOKKOS_INLINE_FUNCTION constexpr store& operator=(
      store<U, UTypes...>&& other) {
    value = FWD(other.value);
    rest  = std::move(other.rest);
    return *this;
  }

  KOKKOS_INLINE_FUNCTION constexpr store& operator=(
      const store& other) noexcept(std::is_nothrow_copy_assignable_v<T> &&
                                   (std::is_nothrow_copy_assignable_v<Types> &&
                                    ...))
    requires(std::is_copy_assignable_v<T>)
  {
    value = other.value;
    rest  = other.rest;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION constexpr store& operator=(store&& other) noexcept(
      std::is_nothrow_move_assignable_v<T> &&
      (std::is_nothrow_move_assignable_v<Types> && ...))
    requires(std::is_move_assignable_v<T>)
  {
    value = std::move(other.value);
    rest  = std::move(other.rest);
    return *this;
  }

#if defined(CEXA_HAS_CXX23)
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(
      const store& other) const
    requires(std::is_copy_assignable_v<const T>)
  {
    value = other.value;
    rest  = other.rest;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(store&& other) const
      noexcept(std::is_nothrow_move_assignable_v<const T> &&
               (std::is_nothrow_move_assignable_v<const Types> && ...))
    requires(std::is_assignable_v<const T&, T &&>)
  {
    value = std::move(other.value);
    rest  = std::move(other.rest);
    return *this;
  }

  template <
      class U, class... UTypes,
      class = std::enable_if_t<sizeof...(UTypes) == sizeof...(Types) &&
                               !(std::is_same_v<U, T> &&
                                 (std::is_same_v<UTypes, Types> && ...)) &&
                               std::is_assignable_v<const T&, const U&>>>
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(
      const store<U, UTypes...>& other) const {
    value = other.value;
    rest  = other.rest;
    return *this;
  }
  template <
      class U, class... UTypes,
      class = std::enable_if_t<sizeof...(UTypes) == sizeof...(Types) &&
                               !(std::is_same_v<U, T> &&
                                 (std::is_same_v<UTypes, Types> && ...)) &&
                               std::is_assignable_v<const T&, U&&>>>
  KOKKOS_INLINE_FUNCTION constexpr const store& operator=(
      store<U, UTypes...>&& other) const {
    value = FWD(other.value);
    rest  = std::move(other.rest);
    return *this;
  }
#endif

  template <std::size_t I>
  KOKKOS_INLINE_FUNCTION constexpr tuple_element_t<I, tuple<T, Types...>>&
  get_value() noexcept {
    if constexpr (I == 0) {
      return value;
    } else {
      return rest.template get_value<I - 1>();
    }
  }

  template <std::size_t I>
  KOKKOS_INLINE_FUNCTION constexpr const tuple_element_t<I, tuple<T, Types...>>&
  get_value() const noexcept {
    if constexpr (I == 0) {
      return value;
    } else {
      return rest.template get_value<I - 1>();
    }
  }

  template <class Type>
  KOKKOS_INLINE_FUNCTION constexpr Type& get_value() noexcept {
    if constexpr (std::is_same_v<Type, T>) {
      return value;
    } else {
      return rest.template get_value<Type>();
    }
  }

  template <class Type>
  KOKKOS_INLINE_FUNCTION constexpr const Type& get_value() const noexcept {
    if constexpr (std::is_same_v<Type, T>) {
      return value;
    } else {
      return rest.template get_value<Type>();
    }
  }

  template <class U, class = std::enable_if_t<std::is_assignable_v<
                         T&, decltype(std::forward<U>(std::declval<U&&>()))>>>
  KOKKOS_INLINE_FUNCTION constexpr void set_all(U&& u) {
    value = u;
  }

  template <class U, class... UTypes,
            class = std::enable_if_t<std::is_assignable_v<
                T&, decltype(std::forward<U>(std::declval<U&&>()))>>>
  KOKKOS_INLINE_FUNCTION constexpr void set_all(U&& head, UTypes&&... tail) {
    value = head;
    rest.set_all(FWD(tail)...);
  }

  template <class UTuple>
  constexpr void set(UTuple&& u) {
    set(FWD(u), std::make_index_sequence<1 + sizeof...(Types)>{});
  }

  template <class UTuple, std::size_t... Ints>
  constexpr void set(UTuple&& u, std::index_sequence<Ints...>) {
    set_all(get<Ints>(FWD(u))...);
  }

  KOKKOS_INLINE_FUNCTION constexpr void swap(store& rhs) noexcept(
      std::is_nothrow_swappable_v<T> &&
      (std::is_nothrow_swappable_v<Types> && ...)) {
    using std::swap;
    swap(value, rhs.value);
    rest.swap(rhs.rest);
  }

#if defined(CEXA_HAS_CXX23)
  KOKKOS_INLINE_FUNCTION constexpr void swap(const store& rhs) const
      noexcept(std::is_nothrow_swappable_v<const T> &&
               (std::is_nothrow_swappable_v<const Types> && ...)) {
    using std::swap;
    swap(value, rhs.value);
    rest.swap(rhs.rest);
  }

#endif

#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
  template <class U, class... UTypes>
    requires std::three_way_comparable_with<T, U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator<=>(
      const store<U, UTypes...>& rhs) const
      -> std::common_comparison_category_t<decltype(value <=> rhs.value),
                                           decltype(rest <=> rhs.rest)> {
    auto res = value <=> rhs.value;
    return res != 0 ? res : rest <=> rhs.rest;
  }
  template <class U, class... UTypes>
  KOKKOS_INLINE_FUNCTION constexpr std::weak_ordering operator<=>(
      const store<U, UTypes...>& rhs) const {
    if (value < rhs.value) {
      return std::weak_ordering::less;
    } else if (rhs.value < value) {
      return std::weak_ordering::greater;
    } else {
      return static_cast<std::weak_ordering>(rest <=> rhs.rest);
    }
  }
  template <class U, class... UTypes>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(
      const store<U, UTypes...>& rhs) const {
    return operator<=>(rhs) == 0;
  }
#else
  template <class U, class... UTypes>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(
      const store<U, UTypes...>& rhs) const {
    return value == rhs.value && rest == rhs.rest;
  }
  template <class U, class... UTypes>
  KOKKOS_INLINE_FUNCTION constexpr bool operator<(
      const store<U, UTypes...>& rhs) const {
    return value < rhs.value || (value == rhs.value && rest < rhs.rest);
  }
#endif
};
}  // namespace impl

template <class... Types>
class tuple;

template <>
class tuple<> {
 public:
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple() noexcept = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr tuple(const tuple& u) noexcept = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple(tuple&& u) noexcept      = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr ~tuple() = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(
      const tuple& u) noexcept = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(tuple&& u) noexcept =
      default;
#if defined(CEXA_HAS_CXX23)
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(
      const tuple& u) const noexcept = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(
      tuple&& u) const noexcept = default;
#endif

  KOKKOS_INLINE_FUNCTION constexpr void swap(tuple&) noexcept {}
#if defined(CEXA_HAS_CXX23)
  KOKKOS_INLINE_FUNCTION constexpr void swap(const tuple&) const noexcept {}
#endif

#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
  KOKKOS_DEFAULTED_FUNCTION auto operator<=>(const tuple&) const = default;
#endif
};

#if !defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
KOKKOS_INLINE_FUNCTION constexpr bool operator==(const tuple<>&,
                                                 const tuple<>&) {
  return true;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const tuple<>&,
                                                 const tuple<>&) {
  return false;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator<(const tuple<>&,
                                                const tuple<>&) {
  return false;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const tuple<>&,
                                                 const tuple<>&) {
  return true;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator>(const tuple<>&,
                                                const tuple<>&) {
  return false;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const tuple<>&,
                                                 const tuple<>&) {
  return true;
}
#endif

template <typename... Types>
// NOTE: move ctor is defined but not detected by clang-tidy, assignment ops are
// intentionally not defined, so that the compiler can default/delete them based
// on what's done inside tuple_assign_helper
// NO_NO_LINT_NEXTLINE(cppcoreguidelines-special-member-functions)
class tuple {
 private:
  template <typename... Ts>
  using T0 = typename impl::nth_type<0, Ts...>::type;
  template <typename... Ts>
  using T1 = typename impl::nth_type<sizeof...(Ts) == 1 ? 0 : 1, Ts...>::type;

  struct converting_tag {};
  struct tuple_like_tag {};

  template <class... UTypes>
  friend class tuple;

  impl::store<Types...> values;

// NOLINTBEGIN(bugprone-macro-parentheses)
#define CONVERTING_TUPLE_CTOR_CONSTRAINTS(CONST, REF)                    \
  class = std::enable_if_t<std::conjunction_v<                           \
      std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,         \
      impl::all_types_constructible<tuple<Types...>,                     \
                                    CONST tuple<UTypes...> REF>,         \
      std::disjunction<                                                  \
          std::bool_constant<sizeof...(Types) != 1>,                     \
          std::negation<std::disjunction<                                \
              std::is_convertible<                                       \
                  decltype(std::declval<CONST tuple<UTypes...> REF>()),  \
                  T0<Types...>>,                                         \
              std::is_constructible<                                     \
                  T0<Types...>,                                          \
                  decltype(std::declval<CONST tuple<UTypes...> REF>())>, \
              std::is_same<T0<Types...>, T0<UTypes...>>>>>,              \
      std::negation<impl::any_types_reference_constructs_from_temporary< \
          tuple<Types...>, CONST tuple<UTypes...> REF>>>>
#define IMPL_CONVERTING_TUPLE_CONSTRUCTOR(CONST, REF)                          \
 public:                                                                       \
  template <                                                                   \
      typename... UTypes,                                                      \
      class = std::enable_if_t<std::conjunction_v<                             \
          std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,           \
          impl::all_types_constructible<tuple<Types...>,                       \
                                        CONST tuple<UTypes...> REF>,           \
          std::disjunction<                                                    \
              std::bool_constant<sizeof...(Types) != 1>,                       \
              std::negation<std::disjunction<                                  \
                  std::is_convertible<                                         \
                      decltype(std::declval<CONST tuple<UTypes...> REF>()),    \
                      T0<Types...>>,                                           \
                  std::is_constructible<                                       \
                      T0<Types...>,                                            \
                      decltype(std::declval<CONST tuple<UTypes...> REF>())>,   \
                  std::is_same<T0<Types...>, T0<UTypes...>>>>>,                \
          std::negation<impl::any_types_reference_constructs_from_temporary<   \
              tuple<Types...>, CONST tuple<UTypes...> REF>>>>>                 \
  KOKKOS_INLINE_FUNCTION explicit(                                             \
      !(impl::all_types_convertible_v<                                         \
          CONST tuple<UTypes...> REF,                                          \
          tuple<Types...>>)) constexpr tuple(CONST tuple<UTypes...> REF other) \
      : tuple(converting_tag{}, FWD(other),                                    \
              std::make_index_sequence<sizeof...(Types)>{}) {}

#define PAIR_CTOR_CONSTRAINTS(CONST, REF)                                  \
  class = std::enable_if_t < sizeof...(Types) == 2 &&                      \
          std::is_constructible_v<                                         \
              T0<Types...>,                                                \
              decltype(std::get<0>(                                        \
                  FWD((std::declval<CONST std::pair<U1, U2> REF>()))))> && \
          std::is_constructible_v<                                         \
              T1<Types...>,                                                \
              decltype(std::get<1>(                                        \
                  FWD((std::declval<CONST std::pair<U1, U2> REF>()))))> && \
          !impl::reference_constructs_from_temporary_v<                    \
              T0<Types...>,                                                \
              decltype(std::get<0>(                                        \
                  FWD((std::declval<CONST std::pair<U1, U2> REF>()))))> && \
          !impl::reference_constructs_from_temporary_v < T1<Types...>,     \
  decltype(std::get<1>(FWD((std::declval<CONST std::pair<U1, U2> REF>())))) >>

#define IMPL_PAIR_CONSTRUCTOR(CONST, REF)                                    \
  template <class U1, class U2,                                              \
            class = std::enable_if_t<                                        \
                sizeof...(Types) == 2 &&                                     \
                std::is_constructible_v<                                     \
                    T0<Types...>,                                            \
                    decltype(std::get<0>(FWD(                                \
                        (std::declval<CONST std::pair<U1, U2> REF>()))))> && \
                std::is_constructible_v<                                     \
                    T1<Types...>,                                            \
                    decltype(std::get<1>(FWD(                                \
                        (std::declval<CONST std::pair<U1, U2> REF>()))))> && \
                !impl::reference_constructs_from_temporary_v<                \
                    T0<Types...>,                                            \
                    decltype(std::get<0>(FWD(                                \
                        (std::declval<CONST std::pair<U1, U2> REF>()))))> && \
                !impl::reference_constructs_from_temporary_v<                \
                    T1<Types...>,                                            \
                    decltype(std::get<1>(FWD(                                \
                        (std::declval<CONST std::pair<U1, U2> REF>()))))>>>  \
  inline constexpr explicit(                                                 \
      (!std::is_convertible_v<                                               \
           decltype(std::get<0>(                                             \
               FWD((std::declval<CONST std::pair<U1, U2> REF>())))),         \
           T0<Types...>> ||                                                  \
       !std::is_convertible_v<                                               \
           decltype(std::get<1>(                                             \
               FWD((std::declval<CONST std::pair<U1, U2> REF>())))),         \
           T1<Types...>>)) tuple(CONST std::pair<U1, U2> REF u)              \
      : values(std::get<0>(FWD(u)), std::get<1>(FWD(u))) {}
  // NOLINTEND(bugprone-macro-parentheses)

  template <class UTuple, std::size_t... Ints>
  KOKKOS_INLINE_FUNCTION constexpr tuple(converting_tag, UTuple&& u,
                                         std::index_sequence<Ints...>)
      : values(get<Ints>(FWD(u))...) {}

  template <class UTuple, std::size_t... Ints>
  constexpr tuple(tuple_like_tag, UTuple&& u, std::index_sequence<Ints...>)
      : values(std::get<Ints>(FWD(u))...) {}

 public:
  // tuple.cnstr
  template <
      class Dummy = void,
      class       = std::enable_if_t<std::conjunction_v<
                std::is_same<Dummy, void>, std::is_default_constructible<Types>...>>>
  KOKKOS_INLINE_FUNCTION explicit(
      (!impl::empty_copy_list_initializable_v<Types> ||
       ...)) constexpr tuple() noexcept((std::
                                             is_nothrow_default_constructible_v<
                                                 Types> &&
                                         ...))
      : values{} {}

  template <
      class Dummy = void,
      class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                     (sizeof...(Types) >= 1 &&
                                (std::is_copy_constructible_v<Types> && ...))>>
  KOKKOS_INLINE_FUNCTION explicit((
      !std::is_convertible_v<const Types&, Types> ||
      ...)) constexpr tuple(const Types&... vals) noexcept((std::
                                                                is_nothrow_copy_constructible_v<
                                                                    Types> &&
                                                            ...))
      : values(vals...) {}

  template <
      class... UTypes,
      class = std::enable_if_t<std::conjunction_v<
          std::bool_constant<sizeof...(Types) == sizeof...(UTypes) &&
                             sizeof...(Types) >= 1>,
          std::negation<
              std::conjunction<std::is_same<UTypes&&, const Types&>...>>,
          std::conditional_t<
              sizeof...(Types) == 1,
              std::negation<std::is_same<std::remove_cvref_t<T0<UTypes...>>,
                                         tuple<Types...>>>,
              std::true_type>,
          std::conditional_t<sizeof...(Types) == 2 || sizeof...(Types) == 3,
                             std::disjunction<
                                 std::negation<std::is_same<
                                     std::remove_cvref_t<T0<UTypes...>>,
                                     std::allocator_arg_t>>,
                                 std::is_same<std::remove_cvref_t<T0<Types...>>,
                                              std::allocator_arg_t>>,
                             std::true_type>,
          std::negation<
              impl::reference_constructs_from_temporary<Types, UTypes&&>>...,
          std::is_constructible<Types, UTypes>...>>>
  KOKKOS_INLINE_FUNCTION explicit((!std::is_convertible_v<UTypes&&, Types> ||
                                   ...)) constexpr tuple(UTypes&&... args)
      : values(FWD(args)...) {}

  KOKKOS_DEFAULTED_FUNCTION constexpr tuple(const tuple& u) = default;
  template <
      class Dummy = void,
      class       = std::enable_if_t<std::is_same_v<Dummy, void> &&
                                     (std::is_move_constructible_v<Types> && ...)>>
  // FIXME: give a reason why we can't mark this one explicit
  // NOLINTNEXTLINE(google-explicit-constructor)
  KOKKOS_INLINE_FUNCTION constexpr tuple(tuple&& u)
      : values(std::move(u.values)) {}

  template <class... UTypes, CONVERTING_TUPLE_CTOR_CONSTRAINTS(const, &)>
  KOKKOS_INLINE_FUNCTION explicit(
      !(impl::all_types_convertible_v<
          const tuple<UTypes...>&,
          tuple<Types...>>)) constexpr tuple(const tuple<UTypes...>& other)
      : tuple(converting_tag{}, FWD(other),
              std::make_index_sequence<sizeof...(Types)>{}) {}

  template <class... UTypes, CONVERTING_TUPLE_CTOR_CONSTRAINTS(, &&)>
  KOKKOS_INLINE_FUNCTION explicit(
      !(impl::all_types_convertible_v<
          tuple<UTypes...>&&,
          tuple<Types...>>)) constexpr tuple(tuple<UTypes...>&& other)
      : tuple(converting_tag{}, std::move(other),
              std::make_index_sequence<sizeof...(Types)>{}) {}

#if defined(CEXA_HAS_CXX23)
  template <class... UTypes, CONVERTING_TUPLE_CTOR_CONSTRAINTS(, &)>
  KOKKOS_INLINE_FUNCTION explicit(
      !(impl::all_types_convertible_v<
          tuple<UTypes...>&,
          tuple<Types...>>)) constexpr tuple(tuple<UTypes...>& other)
      : tuple(converting_tag{}, FWD(other),
              std::make_index_sequence<sizeof...(Types)>{}) {}

  template <class... UTypes, CONVERTING_TUPLE_CTOR_CONSTRAINTS(const, &&)>
  KOKKOS_INLINE_FUNCTION explicit(
      !(impl::all_types_convertible_v<
          const tuple<UTypes...>&&,
          tuple<Types...>>)) constexpr tuple(const tuple<UTypes...>&& other)
      : tuple(converting_tag{}, std::move(other),
              std::make_index_sequence<sizeof...(Types)>{}) {}
#endif
  //   IMPL_CONVERTING_TUPLE_CONSTRUCTOR(const, &)
  //   IMPL_CONVERTING_TUPLE_CONSTRUCTOR(, &&)
  // #if defined(CEXA_HAS_CXX23)
  //   IMPL_CONVERTING_TUPLE_CONSTRUCTOR(, &)
  //   IMPL_CONVERTING_TUPLE_CONSTRUCTOR(const, &&)
  // #endif

  template <class U1, class U2, PAIR_CTOR_CONSTRAINTS(const, &)>
  constexpr explicit(
      (!std::is_convertible_v<decltype(std::get<0>(FWD(
                                  (std::declval<const std::pair<U1, U2>&>())))),
                              T0<Types...>> ||
       !std::is_convertible_v<decltype(std::get<1>(FWD(
                                  (std::declval<const std::pair<U1, U2>&>())))),
                              T1<Types...>>)) tuple(const std::pair<U1, U2>& u)
      : values(u.first, u.second) {}

  template <class U1, class U2, PAIR_CTOR_CONSTRAINTS(, &&)>
  constexpr explicit(
      (!std::is_convertible_v<
           decltype(std::get<0>(FWD((std::declval<std::pair<U1, U2>&&>())))),
           T0<Types...>> ||
       !std::is_convertible_v<
           decltype(std::get<1>(FWD((std::declval<std::pair<U1, U2>&&>())))),
           T1<Types...>>)) tuple(std::pair<U1, U2>&& u)
      : values(std::move(u.first), std::move(u.second)) {
  }  // FIXME: see if we should use forward instead

#if defined(CEXA_HAS_CXX23)
  template <class U1, class U2, PAIR_CTOR_CONSTRAINTS(, &)>
  constexpr explicit(
      (!std::is_convertible_v<
           decltype(std::get<0>(FWD((std::declval<std::pair<U1, U2>&>())))),
           T0<Types...>> ||
       !std::is_convertible_v<
           decltype(std::get<1>(FWD((std::declval<std::pair<U1, U2>&>())))),
           T1<Types...>>)) tuple(std::pair<U1, U2>& u)
      : values(u.first, u.second) {}

  template <class U1, class U2, PAIR_CTOR_CONSTRAINTS(const, &&)>
  constexpr explicit((
      !std::is_convertible_v<decltype(std::get<0>(FWD(
                                 (std::declval<const std::pair<U1, U2>&&>())))),
                             T0<Types...>> ||
      !std::is_convertible_v<decltype(std::get<1>(FWD(
                                 (std::declval<const std::pair<U1, U2>&&>())))),
                             T1<Types...>>)) tuple(const std::pair<U1, U2>&& u)
      : values(std::move(u.first), std::move(u.second)) {
  }  // FIXME: see if we should use forward instead
#endif
  //   IMPL_PAIR_CONSTRUCTOR(const, &)
  //   IMPL_PAIR_CONSTRUCTOR(, &&)
  // #if defined(CEXA_HAS_CXX23)
  //   IMPL_PAIR_CONSTRUCTOR(, &)
  //   IMPL_PAIR_CONSTRUCTOR(const, &&)
  // #endif

  template <
      class UTuple,
      class = std::enable_if_t<std::conjunction_v<
          impl::is_tuple_like<UTuple>,
          std::bool_constant<
              sizeof...(Types) ==
              tuple_size<std::remove_reference_t<UTuple>>::value>,
          impl::all_types_constructible<tuple<Types...>, UTuple&&>,
          std::negation<impl::is_tuple<std::remove_cvref_t<UTuple>>>,
          std::negation<impl::is_subrange<std::remove_cvref_t<UTuple>>>,
          std::negation<impl::any_types_reference_constructs_from_temporary<
              tuple<Types...>, UTuple>>,
          std::conjunction<std::bool_constant<sizeof...(Types) != 1>,
                           std::negation<std::disjunction<
                               std::is_convertible<UTuple, T0<Types...>>,
                               std::is_constructible<T0<Types...>, UTuple>>>>>>>
  constexpr explicit(
      (!impl::all_types_convertible_v<UTuple&&, tuple<Types...>>))
      tuple(UTuple&& u)
      : tuple(tuple_like_tag{}, FWD(u),
              std::make_index_sequence<sizeof...(Types)>{}) {}

  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~tuple() = default;

#undef CONVERTING_TUPLE_CTOR_CONSTRAINTS
#undef PAIR_CTOR_CONSTRAINTS
#undef IMPL_CONVERTING_TUPLE_CONSTRUCTOR
#undef IMPL_PAIR_CONSTRUCTOR

  // tuple.assign
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(const tuple& u)
    requires(std::is_copy_assignable_v<Types> && ...)
  = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(tuple&& u) noexcept(
      (std::is_nothrow_move_assignable_v<Types> && ...))
    requires(std::is_move_assignable_v<Types> && ...)
  = default;

#if defined(CEXA_HAS_CXX23)
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(const tuple& u) const
    requires(std::is_copy_assignable_v<const Types> && ...)
  = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(tuple&& u) const
    requires(std::is_assignable_v<const Types&, Types> && ...)
  = default;
#endif

  template <class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,
                std::is_assignable<Types&, const UTypes&>...>>>
  KOKKOS_INLINE_FUNCTION constexpr tuple&
  operator=(const tuple<UTypes...>& other) noexcept(
      (std::is_nothrow_assignable_v<Types&, const UTypes&> && ...)) {
    values = other.values;
    return *this;
  }

  template <class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,
                std::is_assignable<Types&, UTypes>...>>>
  KOKKOS_INLINE_FUNCTION constexpr tuple&
  operator=(tuple<UTypes...>&& other) noexcept(
      (std::is_nothrow_assignable_v<Types&, UTypes> && ...)) {
    values = std::move(other.values);
    return *this;
  }

#if defined(CEXA_HAS_CXX23)
  template <class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,
                std::is_assignable<const Types&, const UTypes&>...>>>
  KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(
      const tuple<UTypes...>& other) const
      noexcept((std::is_nothrow_assignable_v<const Types&, const UTypes&> &&
                ...)) {
    values = other.values;
    return *this;
  }

  template <class... UTypes,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == sizeof...(UTypes)>,
                std::is_assignable<const Types&, UTypes>...>>>
  KOKKOS_INLINE_FUNCTION constexpr const tuple& operator=(
      tuple<UTypes...>&& other) const
      noexcept((std::is_nothrow_assignable_v<const Types&, UTypes> && ...)) {
    values = std::move(other.values);
    return *this;
  }
#endif

  template <class U1, class U2,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == 2>,
                std::is_assignable<T0<Types...>&, const U1&>,
                std::is_assignable<T1<Types...>&, const U2&>>>>
  constexpr tuple& operator=(const std::pair<U1, U2>& p) noexcept(
      std::is_nothrow_assignable_v<T0<Types...>&, const U1&> &&
      std::is_nothrow_assignable_v<T1<Types...>&, const U2&>) {
    values.value      = p.first;
    values.rest.value = p.second;
    return *this;
  }

  template <class U1, class U2,
            class = std::enable_if_t<
                std::conjunction_v<std::bool_constant<sizeof...(Types) == 2>,
                                   std::is_assignable<T0<Types...>&, U1>,
                                   std::is_assignable<T1<Types...>&, U2>>>>
  // NOTE: we use forward in order to not move out of references contained
  // inside the pair
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  constexpr tuple& operator=(std::pair<U1, U2>&& p) noexcept(
      std::is_nothrow_assignable_v<T0<Types...>&, U1> &&
      std::is_nothrow_assignable_v<T1<Types...>&, U2>) {
    values.value      = FWD(p.first);
    values.rest.value = FWD(p.second);
    return *this;
  }

#if defined(CEXA_HAS_CXX23)
  template <class U1, class U2,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == 2>,
                std::is_assignable<const T0<Types...>&, const U1&>,
                std::is_assignable<const T1<Types...>&, const U2&>>>>
  constexpr const tuple& operator=(const std::pair<U1, U2>& p) const
      noexcept(std::is_nothrow_assignable_v<const T0<Types...>&, const U1&> &&
               std::is_nothrow_assignable_v<const T1<Types...>&, const U2&>) {
    values.value      = p.first;
    values.rest.value = p.second;
    return *this;
  }

  template <class U1, class U2,
            class = std::enable_if_t<std::conjunction_v<
                std::bool_constant<sizeof...(Types) == 2>,
                std::is_assignable<const T0<Types...>&, U1>,
                std::is_assignable<const T1<Types...>&, U2>>>>
  // NOTE: we use forward in order to not move out of references contained
  // inside the pair
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  constexpr const tuple& operator=(std::pair<U1, U2>&& p) const
      noexcept(std::is_nothrow_assignable_v<const T0<Types...>&, U1> &&
               std::is_nothrow_assignable_v<const T1<Types...>&, U2>) {
    values.value      = FWD(p.first);
    values.rest.value = FWD(p.second);
    return *this;
  }
#endif

  // TODO: use conjunction_v here
  // NOTE: This overload is introduced in C++23
  template <class UTuple,
            class = std::enable_if_t<
                impl::is_tuple_like_v<UTuple> &&
                !impl::is_tuple_v<std::remove_cvref_t<UTuple>> &&
                !impl::is_pair<std::remove_cvref_t<UTuple>>::value &&
                impl::is_different_from_v<UTuple, tuple> &&
                !impl::is_subrange_v<UTuple> &&
                sizeof...(Types) ==
                    tuple_size<std::remove_reference_t<UTuple>>::value>>
  // The check for is_assignable is delegated to store.set_all()
  constexpr tuple& operator=(UTuple&& u) {
    values.set(FWD(u));
    return *this;
  }

#if defined(CEXA_HAS_CXX23)
  template <class UTuple,
            class = std::enable_if_t<
                impl::is_tuple_like_v<UTuple> &&
                !impl::is_tuple_v<std::remove_cvref_t<UTuple>> &&
                !impl::is_pair<std::remove_cvref_t<UTuple>>::value &&
                impl::is_different_from_v<UTuple, tuple> &&
                !impl::is_subrange_v<std::remove_cvref_t<UTuple>> &&
                sizeof...(Types) ==
                    tuple_size<std::remove_reference_t<UTuple>>::value>>
  // The check for is_assignable is delegated to store.set_all()
  constexpr const tuple& operator=(UTuple&& u) const {
    values.set(FWD(u));
    return *this;
  }
#endif
#undef FWD

  KOKKOS_INLINE_FUNCTION constexpr void swap(tuple& rhs) noexcept(
      (std::is_nothrow_swappable_v<Types> && ...)) {
    return values.swap(rhs.values);
  }

#if defined(CEXA_HAS_CXX23)
  KOKKOS_INLINE_FUNCTION constexpr void swap(const tuple& rhs) const
      noexcept((std::is_nothrow_swappable_v<const Types> && ...)) {
    return values.swap(rhs.values);
  }
#endif

  template <std::size_t I, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (I < sizeof...(Ts)), typename tuple_element<I, tuple<Ts...>>::type&>
  get(tuple<Ts...>& t) noexcept;
  template <std::size_t I, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (I < sizeof...(Ts)), typename tuple_element<I, tuple<Ts...>>::type&&>
  get(tuple<Ts...>&& t) noexcept;
  template <std::size_t I, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (I < sizeof...(Ts)), const typename tuple_element<I, tuple<Ts...>>::type&>
  get(const tuple<Ts...>& t) noexcept;
  template <std::size_t I, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (I < sizeof...(Ts)),
      const typename tuple_element<I, tuple<Ts...>>::type&&>
  get(const tuple<Ts...>&& t) noexcept;
  template <class T, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (std::is_same_v<T, Ts> || ...), T&>
  get(tuple<Ts...>& t) noexcept;
  template <class T, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (std::is_same_v<T, Ts> || ...), T&&>
  get(tuple<Ts...>&& t) noexcept;
  template <class T, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr std::enable_if_t<
      (std::is_same_v<T, Ts> || ...), const T&>
  get(const tuple<Ts...>& t) noexcept;
  template <class T, class... Ts>
  KOKKOS_INLINE_FUNCTION friend constexpr const std::enable_if_t<
      (std::is_same_v<T, Ts> || ...), const T&&>
  get(const tuple<Ts...>&& t) noexcept;

  // tuple.rel
#if defined(CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR)
  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr auto operator<=>(
      const tuple<UTypes...>& rhs) const {
    return values <=> rhs.values;
  }

  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(
      const tuple<UTypes...>& rhs) const {
    return (values <=> rhs.values) == 0;
  }
#else
  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(
      const tuple<UTypes...>& rhs) const {
    return values == rhs.values;
  }
  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr bool operator!=(
      const tuple<UTypes...>& rhs) const {
    return !(values == rhs.values);
  }
  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr bool operator<(
      const tuple<UTypes...>& rhs) const {
    return values < rhs.values;
  }
  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr bool operator<=(
      const tuple<UTypes...>& rhs) const {
    return !(rhs.values < values);
  }
  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr bool operator>(
      const tuple<UTypes...>& rhs) const {
    return rhs.values < values;
  }
  template <class... UTypes>
    requires(sizeof...(Types) == sizeof...(UTypes))
  KOKKOS_INLINE_FUNCTION constexpr bool operator>=(
      const tuple<UTypes...>& rhs) const {
    return !(values < rhs.values);
  }
#endif
};

// deduction guides
template <class... UTypes>
KOKKOS_DEDUCTION_GUIDE tuple(UTypes...) -> tuple<UTypes...>;
template <class T1, class T2>
tuple(std::pair<T1, T2>) -> tuple<T1, T2>;

// tuple.elem
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)), typename tuple_element<I, tuple<Types...>>::type&>
get(tuple<Types...>& t) noexcept {
  return t.values.template get_value<I>();
}
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)), typename tuple_element<I, tuple<Types...>>::type&&>
// NOTE: this doesn't work with std::move
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
get(tuple<Types...>&& t) noexcept {
  return static_cast<tuple_element_t<I, tuple<Types...>>&&>(
      t.values.template get_value<I>());
}
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)),
    const typename tuple_element<I, tuple<Types...>>::type&>
get(const tuple<Types...>& t) noexcept {
  return t.values.template get_value<I>();
}
template <std::size_t I, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (I < sizeof...(Types)),
    const typename tuple_element<I, tuple<Types...>>::type&&>
get(const tuple<Types...>&& t) noexcept {
  return static_cast<const tuple_element_t<I, tuple<Types...>>&&>(
      t.values.template get_value<I>());
}
template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), T&>
get(tuple<Types...>& t) noexcept {
  return t.values.template get_value<T>();
}
template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), T&&>
// NOTE: this doesn't work with std::move
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
get(tuple<Types...>&& t) noexcept {
  return static_cast<T&&>(t.values.template get_value<T>());
}
template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
    (std::is_same_v<T, Types> || ...), const T&>
get(const tuple<Types...>& t) noexcept {
  return t.values.template get_value<T>();
}
template <class T, class... Types>
KOKKOS_INLINE_FUNCTION constexpr const std::enable_if_t<
    (std::is_same_v<T, Types> || ...), const T&&>
get(const tuple<Types...>&& t) noexcept {
  return static_cast<const T&&>(t.values.template get_value<T>());
}

template <class... Types,
          class = std::enable_if_t<(std::is_swappable_v<Types> && ...)>>
KOKKOS_INLINE_FUNCTION constexpr void swap(
    tuple<Types...>& lhs,
    tuple<Types...>& rhs) noexcept((std::is_nothrow_swappable_v<Types> &&
                                    ...)) {
  lhs.swap(rhs);
}

#if defined(CEXA_HAS_CXX23)
template <class... Types,
          class = std::enable_if_t<(std::is_swappable_v<const Types> && ...)>>
KOKKOS_INLINE_FUNCTION constexpr void
swap(const tuple<Types...>& lhs, const tuple<Types...>& rhs) noexcept(
    (std::is_nothrow_swappable_v<const Types> && ...)) {
  lhs.swap(rhs);
}
#endif

}  // namespace cexa
