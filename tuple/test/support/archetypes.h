//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_ARCHETYPES_H
#define TEST_SUPPORT_ARCHETYPES_H

#include <type_traits>
// #include <cassert>
#include <initializer_list>

#include <Kokkos_Macros.hpp>
#include "cexa_test_macros.hpp"
#include "test_macros.h"
#include "test_workarounds.h"

namespace ArchetypeBases {

template <bool, class T>
struct DepType : T {};

struct NullBase {
#ifndef TEST_WORKAROUND_MSVC_BROKEN_ZA_CTOR_CHECK
protected:
#endif // !TEST_WORKAROUND_MSVC_BROKEN_ZA_CTOR_CHECK
  KOKKOS_DEFAULTED_FUNCTION NullBase() = default;
  KOKKOS_DEFAULTED_FUNCTION NullBase(NullBase const&) = default;
  KOKKOS_DEFAULTED_FUNCTION NullBase& operator=(NullBase const&) = default;
  KOKKOS_DEFAULTED_FUNCTION NullBase(NullBase &&) = default;
  KOKKOS_DEFAULTED_FUNCTION NullBase& operator=(NullBase &&) = default;
};

namespace {
#if defined(CEXA_ON_DEVICE)
template <class D, bool E> __device__ int alive = 0;
template <class D, bool E> __device__ int constructed = 0;
template <class D, bool E> __device__ int value_constructed = 0;
template <class D, bool E> __device__ int default_constructed = 0;
template <class D, bool E> __device__ int copy_constructed = 0;
template <class D, bool E> __device__ int move_constructed = 0;
template <class D, bool E> __device__ int assigned = 0;
template <class D, bool E> __device__ int value_assigned = 0;
template <class D, bool E> __device__ int copy_assigned = 0;
template <class D, bool E> __device__ int move_assigned = 0;
template <class D, bool E> __device__ int destroyed = 0;
#else
template <class D, bool E> int alive = 0;
template <class D, bool E> int constructed = 0;
template <class D, bool E> int value_constructed = 0;
template <class D, bool E> int default_constructed = 0;
template <class D, bool E> int copy_constructed = 0;
template <class D, bool E> int move_constructed = 0;
template <class D, bool E> int assigned = 0;
template <class D, bool E> int value_assigned = 0;
template <class D, bool E> int copy_assigned = 0;
template <class D, bool E> int move_assigned = 0;
template <class D, bool E> int destroyed = 0;
#endif
}

template<class D, bool E>
KOKKOS_INLINE_FUNCTION void reset_constructors() {
  constructed<D, E> = value_constructed<D, E> = default_constructed<D, E> =
    copy_constructed<D, E> = move_constructed<D, E> = 0;
  assigned<D, E> = value_assigned<D, E> = copy_assigned<D, E> = move_assigned<D, E> = destroyed<D, E> = 0;
}

template<class D, bool E>
KOKKOS_INLINE_FUNCTION void reset() {
    CEXA_EXPECT_EQ((alive<D, E>), 0);
    alive<D, E> = 0;
    reset_constructors<D, E>();
}

template <class Derived, bool Explicit = false>
struct TestBase {
    using D = Derived;
    static constexpr bool E = Explicit;

    KOKKOS_INLINE_FUNCTION TestBase() noexcept : value(0) {
        ++alive<D, E>; ++constructed<D, E>; ++default_constructed<D, E>;
    }
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit TestBase(int x) noexcept : value(x) {
        ++alive<D, E>; ++constructed<D, E>; ++value_constructed<D, E>;
    }
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION TestBase(int x) noexcept : value(x) {
        ++alive<D, E>; ++constructed<D, E>; ++value_constructed<D, E>;
    }
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit TestBase(int, int y) noexcept : value(y) {
        ++alive<D, E>; ++constructed<D, E>; ++value_constructed<D, E>;
    }
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION TestBase(int, int y) noexcept : value(y) {
        ++alive<D, E>; ++constructed<D, E>; ++value_constructed<D, E>;
    }
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit TestBase(std::initializer_list<int>& il, int = 0) noexcept
      : value(static_cast<int>(il.size())) {
        ++alive<D, E>; ++constructed<D, E>; ++value_constructed<D, E>;
    }
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit TestBase(std::initializer_list<int>& il, int = 0) noexcept : value(static_cast<int>(il.size())) {
        ++alive<D, E>; ++constructed<D, E>; ++value_constructed<D, E>;
    }
    KOKKOS_INLINE_FUNCTION TestBase& operator=(int xvalue) noexcept {
      value = xvalue;
      ++assigned<D, E>; ++value_assigned<D, E>;
      return *this;
    }
#ifndef TEST_WORKAROUND_MSVC_BROKEN_ZA_CTOR_CHECK
protected:
#endif // !TEST_WORKAROUND_MSVC_BROKEN_ZA_CTOR_CHECK
    KOKKOS_INLINE_FUNCTION ~TestBase() {
      CEXA_EXPECT(value != -999); CEXA_EXPECT((alive<D, E>) > 0);
      --alive<D, E>; ++destroyed<D, E>; value = -999;
    }
    KOKKOS_INLINE_FUNCTION explicit TestBase(TestBase const& o) noexcept : value(o.value) {
        CEXA_EXPECT(o.value != -1); CEXA_EXPECT(o.value != -999);
        ++alive<D, E>; ++constructed<D, E>; ++copy_constructed<D, E>;
    }
    KOKKOS_INLINE_FUNCTION explicit TestBase(TestBase && o) noexcept : value(o.value) {
        CEXA_EXPECT(o.value != -1); CEXA_EXPECT(o.value != -999);
        ++alive<D, E>; ++constructed<D, E>; ++move_constructed<D, E>;
        o.value = -1;
    }
    KOKKOS_INLINE_FUNCTION TestBase& operator=(TestBase const& o) noexcept {
      CEXA_EXPECT(o.value != -1); CEXA_EXPECT(o.value != -999);
      ++assigned<D, E>; ++copy_assigned<D, E>;
      value = o.value;
      return *this;
    }
    KOKKOS_INLINE_FUNCTION TestBase& operator=(TestBase&& o) noexcept {
        CEXA_EXPECT(o.value != -1); CEXA_EXPECT(o.value != -999);
        ++assigned<D, E>; ++move_assigned<D, E>;
        value = o.value;
        o.value = -1;
        return *this;
    }
public:
    int value;
};

template <bool Explicit = false>
struct ValueBase {
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit constexpr ValueBase(int x) : value(x) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION constexpr ValueBase(int x) : value(x) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit constexpr ValueBase(int, int y) : value(y) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION constexpr ValueBase(int, int y) : value(y) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit constexpr ValueBase(std::initializer_list<int>& il, int = 0) : value(static_cast<int>(il.size())) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION constexpr ValueBase(std::initializer_list<int>& il, int = 0) : value(static_cast<int>(il.size())) {}
    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 ValueBase& operator=(int xvalue) noexcept {
        value = xvalue;
        return *this;
    }
    //~ValueBase() { assert(value != -999); value = -999; }
    int value;
#ifndef TEST_WORKAROUND_MSVC_BROKEN_ZA_CTOR_CHECK
protected:
#endif // !TEST_WORKAROUND_MSVC_BROKEN_ZA_CTOR_CHECK
    KOKKOS_INLINE_FUNCTION constexpr static int check_value(int const& val) {
      CEXA_EXPECT(val != -1); CEXA_EXPECT(val != 999);
      return val;
    }
    KOKKOS_INLINE_FUNCTION constexpr static int check_value(int& val, int val_cp = 0) {
      CEXA_EXPECT(val != -1); CEXA_EXPECT(val != 999);
      val_cp = val;
      val = -1;
      return val_cp;
    }
    KOKKOS_INLINE_FUNCTION constexpr ValueBase() noexcept : value(0) {}
    KOKKOS_INLINE_FUNCTION constexpr ValueBase(ValueBase const& o) noexcept : value(check_value(o.value)) {
    }
    KOKKOS_INLINE_FUNCTION constexpr ValueBase(ValueBase && o) noexcept : value(check_value(o.value)) {
    }
    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 ValueBase& operator=(ValueBase const& o) noexcept {
        CEXA_EXPECT(o.value != -1); CEXA_EXPECT(o.value != -999);
        value = o.value;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 ValueBase& operator=(ValueBase&& o) noexcept {
        CEXA_EXPECT(o.value != -1); CEXA_EXPECT(o.value != -999);
        value = o.value;
        o.value = -1;
        return *this;
    }
};


template <bool Explicit = false>
struct TrivialValueBase {
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit constexpr TrivialValueBase(int x) : value(x) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION constexpr TrivialValueBase(int x) : value(x) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit constexpr TrivialValueBase(int, int y) : value(y) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION constexpr TrivialValueBase(int, int y) : value(y) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION explicit constexpr TrivialValueBase(std::initializer_list<int>& il, int = 0) : value(static_cast<int>(il.size())) {}
    template <bool Dummy = true, typename std::enable_if<Dummy && !Explicit, bool>::type = true>
    KOKKOS_INLINE_FUNCTION constexpr TrivialValueBase(std::initializer_list<int>& il, int = 0) : value(static_cast<int>(il.size())) {}
    int value;
#ifndef TEST_WORKAROUND_MSVC_BROKEN_ZA_CTOR_CHECK
protected:
#endif // !TEST_WORKAROUND_MSVC_BROKEN_ZA_CTOR_CHECK
    KOKKOS_INLINE_FUNCTION constexpr TrivialValueBase() noexcept : value(0) {}
};

}

//============================================================================//
// Trivial Implicit Test Types
namespace ImplicitTypes {
#include "archetypes.ipp"
}

//============================================================================//
// Trivial Explicit Test Types
namespace ExplicitTypes {
#define DEFINE_EXPLICIT explicit
#include "archetypes.ipp"
}

//============================================================================//
//
namespace NonConstexprTypes {
#define DEFINE_CONSTEXPR
#include "archetypes.ipp"
}

//============================================================================//
// Non-literal implicit test types
namespace NonLiteralTypes {
#define DEFINE_ASSIGN_CONSTEXPR
#define DEFINE_DTOR(Name) KOKKOS_INLINE_FUNCTION ~Name() {}
#include "archetypes.ipp"
}

//============================================================================//
// Non-throwing implicit test types
namespace NonThrowingTypes {
#define DEFINE_NOEXCEPT noexcept
#include "archetypes.ipp"
}

//============================================================================//
// Non-Trivially Copyable Implicit Test Types
namespace NonTrivialTypes {
#define DEFINE_CTOR {}
#define DEFINE_ASSIGN { return *this; }
#include "archetypes.ipp"
}

//============================================================================//
// Implicit counting types
namespace TestTypes {
#define DEFINE_CONSTEXPR
#define DEFINE_BASE(Name) ::ArchetypeBases::TestBase<Name>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

}

//============================================================================//
// Implicit counting types
namespace ExplicitTestTypes {
#define DEFINE_CONSTEXPR
#define DEFINE_EXPLICIT explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::TestBase<Name, true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

}

//============================================================================//
// Implicit value types
namespace ConstexprTestTypes {
#define DEFINE_BASE(Name) ::ArchetypeBases::ValueBase<>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

} // namespace ConstexprTestTypes


//============================================================================//
//
namespace ExplicitConstexprTestTypes {
#define DEFINE_EXPLICIT explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::ValueBase<true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

} // namespace ExplicitConstexprTestTypes


//============================================================================//
//
namespace TrivialTestTypes {
#define DEFINE_BASE(Name) ::ArchetypeBases::TrivialValueBase<false>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

} // namespace TrivialTestTypes

//============================================================================//
//
namespace ExplicitTrivialTestTypes {
#define DEFINE_EXPLICIT explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::TrivialValueBase<true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

} // namespace ExplicitTrivialTestTypes

#endif // TEST_SUPPORT_ARCHETYPES_H
