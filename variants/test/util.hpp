// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)
// SPDX-FileCopyrightText: Michael Park
// SPDX-License-Identifier: BSL-1.0

#ifndef _UTIL_HPP
#define _UTIL_HPP

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if (__has_feature(cxx_exceptions) || defined(__cpp_exceptions) ||             \
     (defined(_MSC_VER) && defined(_CPPUNWIND)) || defined(__EXCEPTIONS)) &&   \
    !(defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_HIP) ||             \
      defined(KOKKOS_ENABLE_CUDA))

#define EXCEPTIONS_AVAILABLE
#endif

#include <Kokkos_Core.hpp>
#include <mpark/config.hpp>

#include <Kokkos_Variant.hpp>

enum Qual { Ptr, ConstPtr, LRef, ConstLRef, RRef, ConstRRef };

struct get_qual {
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

#ifdef EXCEPTIONS_AVAILABLE
struct CopyConstruction : std::exception {};
struct CopyAssignment : std::exception {};
struct MoveConstruction : std::exception {};
struct MoveAssignment : std::exception {};

struct copy_thrower_t {
  KOKKOS_FUNCTION constexpr copy_thrower_t() {}
  KOKKOS_FUNCTION [[noreturn]] copy_thrower_t(const copy_thrower_t &) {
    throw CopyConstruction{};
  }
  copy_thrower_t(copy_thrower_t &&) = default;
  KOKKOS_FUNCTION copy_thrower_t &operator=(const copy_thrower_t &) {
    throw CopyAssignment{};
  }
  copy_thrower_t &operator=(copy_thrower_t &&) = default;
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
  move_thrower_t(const move_thrower_t &) = default;
  KOKKOS_FUNCTION [[noreturn]] move_thrower_t(move_thrower_t &&) {
    throw MoveConstruction{};
  }
  move_thrower_t &operator=(const move_thrower_t &) = default;
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

namespace test_util {
// We need an object with property similar to std::string but available on any
// device, so we badly reimplement the features we need
class DeviceString {
  char *_data;
  size_t _size;
  size_t _capacity;

  KOKKOS_FUNCTION void allocate_and_copy(const char *data, size_t size) {
    _size     = size;
    _capacity = _size + 1;
    _data     = static_cast<char *>(malloc(_capacity * sizeof(char)));
    Kokkos::Impl::strcpy(_data, data);
  }

  KOKKOS_FUNCTION void allocate_and_copy(const char *data) {
    allocate_and_copy(data, Kokkos::Impl::strlen(data));
  }

  // Convert rhs to DeviceString in base 10
  template <typename T>
  KOKKOS_FUNCTION void int_to_string(T rhs) {
    // Find size of string
    T remainder = rhs;
    _size       = 0;
    do {
      remainder /= 10;
      ++_size;
    } while (remainder != 0);

    // Allocate
    _capacity = _size + 1;
    _data     = static_cast<char *>(malloc(sizeof(char) * _capacity));

    // Fill up the string
    remainder = rhs;
    int i     = _size;
    _data[i]  = '\0';
    do {
      _data[--i] = '0' + (remainder % 10);
      remainder /= 10;
    } while (remainder != 0);
  }

 public:
  // Destructor
  KOKKOS_FUNCTION ~DeviceString() { free(_data); }

  // Constructors
  KOKKOS_FUNCTION DeviceString() { allocate_and_copy("", 0); }

  KOKKOS_FUNCTION DeviceString(const char *data) { allocate_and_copy(data); }

  KOKKOS_FUNCTION DeviceString(const DeviceString &rhs) {
    allocate_and_copy(rhs._data, rhs._size);
  }

  KOKKOS_FUNCTION DeviceString(const std::initializer_list<char> ilist) {
    _size      = ilist.size();
    _capacity  = _size + 1;
    _data      = static_cast<char *>(malloc(_capacity * sizeof(char)));
    char *data = _data;
    for (const auto &c : ilist) {
      *(data++) = c;
    }
    *data = '\0';
  }

  KOKKOS_FUNCTION DeviceString(int rhs) { int_to_string<int>(rhs); }

  KOKKOS_FUNCTION DeviceString(long rhs) { int_to_string<long>(rhs); }

  KOKKOS_FUNCTION DeviceString(short rhs) { int_to_string<short>(rhs); }

  KOKKOS_FUNCTION DeviceString(char rhs) {
    char tmp[2] = {rhs, '\0'};
    allocate_and_copy(tmp, 1);
  }

  KOKKOS_FUNCTION DeviceString(double) {
    // Needs to be defined for the tests to compile
    Kokkos::abort(
        "Conversion from double/float to DeviceString is not implemented");
  }

  KOKKOS_FUNCTION DeviceString(DeviceString &&other) noexcept {
    _capacity = _size = 0;
    _data             = nullptr;
    Kokkos::kokkos_swap(_capacity, other._capacity);
    Kokkos::kokkos_swap(_data, other._data);
    Kokkos::kokkos_swap(_size, other._size);
  }

  // Getter
  KOKKOS_FUNCTION constexpr size_t capacity() const { return _capacity; }

  // Affectation operators
  KOKKOS_FUNCTION DeviceString &operator=(const char *rhs) {
    _size = Kokkos::Impl::strlen(rhs);
    if (_capacity >= _size + 1) {
      Kokkos::Impl::strcpy(_data, rhs);
    } else {
      free(_data);
      allocate_and_copy(rhs, _size);
    }

    return *this;
  }

  KOKKOS_FUNCTION DeviceString &operator=(const DeviceString &other) {
    if (this == &other) {
      return *this;
    } else if (_capacity >= other._size + 1) {
      Kokkos::Impl::strcpy(_data, other._data);
      _size = other._size;
    } else {
      DeviceString temp(other);
      Kokkos::kokkos_swap(_data, temp._data);
      Kokkos::kokkos_swap(_capacity, temp._capacity);
      Kokkos::kokkos_swap(_size, temp._size);
    }

    return *this;
  }

  KOKKOS_FUNCTION DeviceString &operator=(DeviceString &&other) noexcept {
    DeviceString temp(std::move(other));
    Kokkos::kokkos_swap(_data, temp._data);
    Kokkos::kokkos_swap(_capacity, temp._capacity);
    Kokkos::kokkos_swap(_size, temp._size);
    return *this;
  }

  // Comparison operators
  KOKKOS_FUNCTION constexpr bool operator==(const DeviceString &rhs) const {
    if (rhs._size == _size) {
      return !Kokkos::Impl::strcmp(rhs._data, this->_data);
    } else {
      return false;
    }
  }

  KOKKOS_FUNCTION constexpr bool operator!=(const DeviceString &rhs) const {
    return !(this->operator==(rhs));
  }

  friend KOKKOS_FUNCTION constexpr bool operator!=(const char *lhs,
                                                   const DeviceString &rhs);
  friend KOKKOS_FUNCTION constexpr bool operator==(const char *lhs,
                                                   const DeviceString &rhs);
  friend KOKKOS_FUNCTION constexpr bool operator!=(const DeviceString &lhs,
                                                   const char *rhs);
  friend KOKKOS_FUNCTION constexpr bool operator==(const DeviceString &lhs,
                                                   const char *rhs);

  // Concatenation
  KOKKOS_FUNCTION DeviceString &operator+=(const DeviceString &rhs) {
    _capacity      = _size + rhs._size + 1;
    char *tmp_data = static_cast<char *>(malloc(sizeof(char) * _capacity));

    Kokkos::Impl::strcpy(tmp_data, _data);
    Kokkos::Impl::strcpy(tmp_data + _size, rhs._data);

    free(_data);
    _data = tmp_data;
    _size += rhs._size;

    return *this;
  }

  KOKKOS_FUNCTION DeviceString operator+(const DeviceString &rhs) const {
    DeviceString tmp(*this);
    tmp += rhs;
    return tmp;
  }
};

KOKKOS_FUNCTION constexpr bool operator!=(const char *lhs,
                                          const DeviceString &rhs) {
  return Kokkos::Impl::strcmp(lhs, rhs._data);
}
KOKKOS_FUNCTION constexpr bool operator==(const char *lhs,
                                          const DeviceString &rhs) {
  return !operator!=(lhs, rhs);
}
KOKKOS_FUNCTION constexpr bool operator!=(const DeviceString &lhs,
                                          const char *rhs) {
  return operator!=(rhs, lhs);
}
KOKKOS_FUNCTION constexpr bool operator==(const DeviceString &lhs,
                                          const char *rhs) {
  return operator==(rhs, lhs);
}
}  // namespace test_util

// Helper function for test
template <typename T>
void test_helper() {
  int errors = 0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy(0, 1), T{}, errors);
  EXPECT_EQ(0, errors);
}

// Dumbed down version of gtest's EXPECT_ functions, usable on the device
// (needs to reduce on an argument named `error`)
#define DEXPECT_EQ(arg1, arg2)    \
  do {                            \
    error += !((arg1) == (arg2)); \
  } while (false);

#define DEXPECT_NE(arg1, arg2)    \
  do {                            \
    error += !((arg1) != (arg2)); \
  } while (false);

#define DEXPECT_TRUE(arg) \
  do {                    \
    error += !(arg);      \
  } while (false);

#define DEXPECT_FALSE(arg) \
  do {                     \
    error += !!(arg);      \
  } while (false);

// main function: init Kokkos, init GTest, lauch tests
#define TEST_MAIN                           \
  int main(int argc, char *argv[]) {        \
    Kokkos::ScopeGuard kokkos(argc, argv);  \
    ::testing::InitGoogleTest(&argc, argv); \
                                            \
    int result = RUN_ALL_TESTS();           \
    return result;                          \
  }

#endif
