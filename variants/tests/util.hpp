// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)
// SPDX-FileCopyrightText: Michael Park
// SPDX-License-Identifier: BSL-1.0

#ifndef KOKKOS_VARIANT_TEST_UTIL_HPP
#define KOKKOS_VARIANT_TEST_UTIL_HPP

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if (__has_feature(cxx_exceptions) || defined(__cpp_exceptions) ||           \
     (defined(_MSC_VER) && defined(_CPPUNWIND)) || defined(__EXCEPTIONS)) && \
    !(defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_HIP) ||           \
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

// By default, HIP compiler tries to inline every function, which makes
// compilation of some tests extremly slow.
#ifdef KOKKOS_ENABLE_HIP
#define KOKKOS_HIP_NO_INLINE __attribute__((noinline))
#else
#define KOKKOS_HIP_NO_INLINE
#endif

namespace test_util {
// We need a way to allocate memory from the device, other backends are fine
// but SYCL needs its own solution, it is only an emulation of memory
// management.
struct MemPool {
#ifdef KOKKOS_ENABLE_SYCL
  static const uint8_t poolsize = 50;
  uint8_t mempool[poolsize];

  KOKKOS_FUNCTION void *malloc(size_t size) { return mempool; }
  KOKKOS_FUNCTION void free(void *ptr) {}
#else
  KOKKOS_FUNCTION void *malloc(size_t size) { return ::malloc(size); }
  KOKKOS_FUNCTION void free(void *ptr) { ::free(ptr); }
#endif
};

// We need an object with property similar to std::string but available on any
// device, so we badly reimplement the features we need
class DeviceString {
  MemPool _mempool;

  char *_data;
  size_t _size;
  size_t _capacity;

  KOKKOS_FUNCTION void allocate_and_copy(const char *data, size_t size) {
    _size     = size;
    _capacity = _size + 1;
    _data     = static_cast<char *>(_mempool.malloc(_capacity * sizeof(char)));
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
    if (rhs < 0) {
      // Extra space to store the '-'
      ++_size;
    }
    do {
      remainder /= 10;
      ++_size;
    } while (remainder != 0);

    // Allocate
    _capacity = _size + 1;
    _data     = static_cast<char *>(_mempool.malloc(sizeof(char) * _capacity));

    // Fill up the string
    remainder = rhs;
    int i     = _size;
    _data[i]  = '\0';
    if (rhs < 0) {
      _data[0]  = '-';
      remainder = -rhs;
    }
    do {
      _data[--i] = '0' + (remainder % 10);
      remainder /= 10;
    } while (remainder != 0);
  }

 public:
  // Destructor
  KOKKOS_FUNCTION ~DeviceString() { _mempool.free(_data); }

  // Constructors
  KOKKOS_FUNCTION DeviceString() { allocate_and_copy("", 0); }

  KOKKOS_FUNCTION DeviceString(const char *data) { allocate_and_copy(data); }

  KOKKOS_FUNCTION DeviceString(const DeviceString &rhs) {
    allocate_and_copy(rhs._data, rhs._size);
  }

  KOKKOS_FUNCTION DeviceString(const std::initializer_list<char> ilist) {
    _size      = ilist.size();
    _capacity  = _size + 1;
    _data      = static_cast<char *>(_mempool.malloc(_capacity * sizeof(char)));
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
#ifdef KOKKOS_ENABLE_SYCL
    _data =
        static_cast<char *>(_mempool.malloc(other._capacity * sizeof(char)));
    Kokkos::Impl::strcpy(_data, other._data);
#else
    _data = nullptr;
    Kokkos::kokkos_swap(_data, other._data);
#endif
    Kokkos::kokkos_swap(_capacity, other._capacity);
    Kokkos::kokkos_swap(_size, other._size);
  }

  // Getter
  KOKKOS_FUNCTION constexpr size_t capacity() const { return _capacity; }

  KOKKOS_FUNCTION constexpr char *c_str() const { return _data; }

  // Affectation operators
  KOKKOS_FUNCTION DeviceString &operator=(const char *rhs) {
    _size = Kokkos::Impl::strlen(rhs);
    if (_capacity >= _size + 1) {
      Kokkos::Impl::strcpy(_data, rhs);
    } else {
      _mempool.free(_data);
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
#ifdef KOKKOS_ENABLE_SYCL
      Kokkos::Impl::strcpy(_data, temp._data);
#else
      Kokkos::kokkos_swap(_data, temp._data);
#endif
      Kokkos::kokkos_swap(_capacity, temp._capacity);
      Kokkos::kokkos_swap(_size, temp._size);
    }

    return *this;
  }

  KOKKOS_FUNCTION DeviceString &operator=(DeviceString &&other) noexcept {
    DeviceString temp(std::move(other));
#ifdef KOKKOS_ENABLE_SYCL
    Kokkos::Impl::strcpy(_data, temp._data);
#else
    Kokkos::kokkos_swap(_data, temp._data);
#endif
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
    _capacity = _size + rhs._size + 1;
    // It still works on Sycl despite tmp_data == _data
    char *tmp_data =
        static_cast<char *>(_mempool.malloc(sizeof(char) * _capacity));

    Kokkos::Impl::strcpy(tmp_data, _data);
    Kokkos::Impl::strcpy(tmp_data + _size, rhs._data);

    _mempool.free(_data);
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

// Helper function for test: launch the test on both host and device
template <typename T>
void test_helper() {
  int num_errors = 0;
  // Execute on device
  Kokkos::parallel_reduce(Kokkos::RangePolicy(0, 1), T{}, num_errors);
  EXPECT_EQ(0, num_errors);

  // Execute on host if different
  if constexpr (!std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                                Kokkos::DefaultExecutionSpace>) {
    num_errors = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, 1), T{},
        num_errors);
    EXPECT_EQ(0, num_errors);
  }
}

// Print the value of t deducing a sensible format from its type T.
template <typename T>
KOKKOS_FUNCTION void print_arg_value(T t) {
  using type = std::remove_reference_t<std::remove_cv_t<T>>;
  if constexpr (std::is_same_v<T, char *> || std::is_same_v<T, const char *>) {
    if (t != nullptr) {
      // FIXME This is not 100% safe, as there is no guarantee that a char* is
      // a pointer to a C-style string
      Kokkos::printf("\"%s\"", t);
    } else {
      Kokkos::printf("NULL");
    }
  } else if constexpr (std::is_pointer_v<type>) {
    if (t != nullptr) {
      Kokkos::printf("%p", t);
    } else {
      Kokkos::printf("NULL");
    }
  } else if constexpr (std::is_same_v<type, bool>) {
    if (t) {
      Kokkos::printf("true");
    } else {
      Kokkos::printf("false");
    }
  } else if constexpr (std::is_integral_v<type>) {
    if constexpr (std::is_unsigned_v<type>) {
      Kokkos::printf("%u", t);
    } else {
      Kokkos::printf("%i", t);
    }
  } else if constexpr (std::is_floating_point_v<type>) {
    Kokkos::printf("%f", t);
  } else if constexpr (std::is_same_v<type, DeviceString>) {
    Kokkos::printf("\"%s\"", t.c_str());
  } else if constexpr (std::is_enum_v<type>) {
    Kokkos::printf("%i", t);
  } else {
    Kokkos::printf("Unknown type, can't display value.");
  }
}
}  // namespace test_util

// Dumbed down version of gtest's EXPECT_ functions, usable on the device
// (needs to reduce on an argument named `errors`)
// FIXME the printf can get interleaved when executing in parallel, the whole
// error message should be constructed at once in a string and displayed in a
// printf, but there is no robust enough implementation of string on GPU to do
// that portably
#define DEXPECT_EQ(arg1, arg2)                                               \
  do {                                                                       \
    if (!((arg1) == (arg2))) {                                               \
      errors += 1;                                                           \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__);                \
      Kokkos::printf("Expected equality of these values:\n");                \
      Kokkos::printf("  " KOKKOS_IMPL_STRINGIFY(arg1) "\n    Which is: ");   \
      test_util::print_arg_value(arg1);                                      \
      Kokkos::printf("\n  " KOKKOS_IMPL_STRINGIFY(arg2) "\n    Which is: "); \
      test_util::print_arg_value(arg2);                                      \
      Kokkos::printf("\n");                                                  \
    }                                                                        \
  } while (false)

#define DEXPECT_NE(arg1, arg2)                                       \
  do {                                                               \
    if (!((arg1) != (arg2))) {                                       \
      errors += 1;                                                   \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__);        \
      Kokkos::printf("Expected: (" KOKKOS_IMPL_STRINGIFY(            \
          arg1) ") != (" KOKKOS_IMPL_STRINGIFY(arg2) "), actual: "); \
      test_util::print_arg_value(arg1);                              \
      Kokkos::printf(" vs ");                                        \
      test_util::print_arg_value(arg2);                              \
      Kokkos::printf("\n");                                          \
    }                                                                \
  } while (false)

#define DEXPECT_TRUE(arg)                                     \
  do {                                                        \
    if (!(arg)) {                                             \
      errors += 1;                                            \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__); \
      Kokkos::printf("Value of: " KOKKOS_IMPL_STRINGIFY(      \
          arg) "\n  Actual: false\nExpected: true\n");        \
    }                                                         \
  } while (false)

#define DEXPECT_FALSE(arg)                                    \
  do {                                                        \
    if (!!(arg)) {                                            \
      errors += 1;                                            \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__); \
      Kokkos::printf("Value of: " KOKKOS_IMPL_STRINGIFY(      \
          arg) "\n  Actual: true\nExpected: false\n");        \
    }                                                         \
  } while (false)

// main function: init Kokkos, init GTest, lauch tests
#define TEST_MAIN                           \
  int main(int argc, char *argv[]) {        \
    Kokkos::ScopeGuard kokkos(argc, argv);  \
    ::testing::InitGoogleTest(&argc, argv); \
                                            \
    int result = RUN_ALL_TESTS();           \
    return result;                          \
  }

#endif  // KOKKOS_UTIL_HPP
