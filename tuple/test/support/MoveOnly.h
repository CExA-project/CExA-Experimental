//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MOVEONLY_H
#define MOVEONLY_H

#include "test_macros.h"

#include <cstddef>
#include <functional>

#include <Kokkos_Macros.hpp>

class MoveOnly
{
    int data_;
public:
    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR MoveOnly(int data = 1) : data_(data) {}

    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 MoveOnly(MoveOnly&& x) TEST_NOEXCEPT
        : data_(x.data_) {x.data_ = 0;}
    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR int get() const {return data_;}

    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator==(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ == y.data_; }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator!=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ != y.data_; }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator< (const MoveOnly& x, const MoveOnly& y)
        { return x.data_ <  y.data_; }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator<=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ <= y.data_; }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator> (const MoveOnly& x, const MoveOnly& y)
        { return x.data_ >  y.data_; }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator>=(const MoveOnly& x, const MoveOnly& y)
        { return x.data_ >= y.data_; }

#if TEST_STD_VER > 17
    KOKKOS_INLINE_FUNCTION friend constexpr auto operator<=>(const MoveOnly&, const MoveOnly&) = default;
#endif // TEST_STD_VER > 17

    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 MoveOnly operator+(const MoveOnly& x) const
        { return MoveOnly(data_ + x.data_); }
    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 MoveOnly operator*(const MoveOnly& x) const
        { return MoveOnly(data_ * x.data_); }

    template<class T>
    friend void operator,(MoveOnly const&, T) = delete;

    template<class T>
    friend void operator,(T, MoveOnly const&) = delete;

  // FIXME: The default std::swap seems to not work on on cuda 12.2-12.4 + gcc 11.2-12.2
#if defined(KOKKOS_COMPILER_NVCC) && defined(KOKKOS_COMPILER_GNU) && KOKKOS_COMPILER_GNU < 1300
    KOKKOS_INLINE_FUNCTION friend constexpr void swap(MoveOnly& lhs, MoveOnly& rhs) {
        auto tmp = lhs.data_;
        lhs.data_ = rhs.data_;
        rhs.data_ = tmp;
    }
#endif
};

template <>
struct std::hash<MoveOnly>
{
    typedef MoveOnly argument_type;
    typedef std::size_t result_type;
    TEST_CONSTEXPR std::size_t operator()(const MoveOnly& x) const {return static_cast<size_t>(x.get());}
};

class TrivialMoveOnly {
    int data_;

  public:
    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR TrivialMoveOnly(int data = 1) : data_(data) {}

    TrivialMoveOnly(const TrivialMoveOnly&)            = delete;
    TrivialMoveOnly& operator=(const TrivialMoveOnly&) = delete;

    KOKKOS_DEFAULTED_FUNCTION TrivialMoveOnly(TrivialMoveOnly&&)            = default;
    KOKKOS_DEFAULTED_FUNCTION TrivialMoveOnly& operator=(TrivialMoveOnly&&) = default;

    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR int get() const { return data_; }

    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator==(const TrivialMoveOnly& x, const TrivialMoveOnly& y) {
      return x.data_ == y.data_;
    }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator!=(const TrivialMoveOnly& x, const TrivialMoveOnly& y) {
      return x.data_ != y.data_;
    }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator<(const TrivialMoveOnly& x, const TrivialMoveOnly& y) {
      return x.data_ < y.data_;
    }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator<=(const TrivialMoveOnly& x, const TrivialMoveOnly& y) {
      return x.data_ <= y.data_;
    }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator>(const TrivialMoveOnly& x, const TrivialMoveOnly& y) {
      return x.data_ > y.data_;
    }
    KOKKOS_INLINE_FUNCTION friend TEST_CONSTEXPR bool operator>=(const TrivialMoveOnly& x, const TrivialMoveOnly& y) {
      return x.data_ >= y.data_;
    }

#if TEST_STD_VER > 17
    KOKKOS_INLINE_FUNCTION friend constexpr auto operator<=>(const TrivialMoveOnly&, const TrivialMoveOnly&) = default;
#endif // TEST_STD_VER > 17

    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 TrivialMoveOnly operator+(const TrivialMoveOnly& x) const {
      return TrivialMoveOnly(data_ + x.data_);
    }
    KOKKOS_INLINE_FUNCTION TEST_CONSTEXPR_CXX14 TrivialMoveOnly operator*(const TrivialMoveOnly& x) const {
      return TrivialMoveOnly(data_ * x.data_);
    }

    template<class T>
    friend void operator,(TrivialMoveOnly const&, T) = delete;

    template<class T>
    friend void operator,(T, TrivialMoveOnly const&) = delete;
};

template <>
struct std::hash<TrivialMoveOnly> {
    typedef TrivialMoveOnly argument_type;
    typedef std::size_t result_type;
    TEST_CONSTEXPR std::size_t operator()(const TrivialMoveOnly& x) const { return static_cast<size_t>(x.get()); }
};

#endif // MOVEONLY_H
