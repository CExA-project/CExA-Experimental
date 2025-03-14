// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)
// SPDX-FileCopyrightText: Michael Park
// SPDX-License-Identifier: BSL-1.0

#ifndef MPARK_CONFIG_HPP
#define MPARK_CONFIG_HPP

#include <Kokkos_Core.hpp>

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#ifndef __has_include
#define __has_include(x) 0
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_builtin(__builtin_addressof) ||                                      \
    (defined(__GNUC__) && __GNUC__ >= 7) || defined(_MSC_VER)
#define MPARK_BUILTIN_ADDRESSOF
#endif

#if __has_builtin(__builtin_unreachable) || defined(__GNUC__)
#define MPARK_BUILTIN_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define MPARK_BUILTIN_UNREACHABLE __assume(false)
#else
#define MPARK_BUILTIN_UNREACHABLE
#endif

#if __has_builtin(__type_pack_element) && !(defined(__ICC))
#define MPARK_TYPE_PACK_ELEMENT
#endif

#if (__has_feature(cxx_exceptions) || defined(__cpp_exceptions) ||             \
     (defined(_MSC_VER) && defined(_CPPUNWIND)) || defined(__EXCEPTIONS)) &&   \
    !(defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_HIP) ||             \
      defined(KOKKOS_ENABLE_CUDA))
#define MPARK_EXCEPTIONS
#endif

#if !defined(__GLIBCXX__) || __has_include(<codecvt>) // >= libstdc++-5
#define MPARK_TRIVIALITY_TYPE_TRAITS
#define MPARK_INCOMPLETE_TYPE_TRAITS
#endif

#endif // MPARK_CONFIG_HPP
