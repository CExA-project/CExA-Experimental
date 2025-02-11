// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <map>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "util.hpp"

// This issues are specific to host specific function so no need to test them on
// GPU
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) &&             \
    !defined(KOKKOS_ENABLE_SYCL)

#ifdef MPARK_INCOMPLETE_TYPE_TRAITS
// https://github.com/mpark/variant/issues/34
TEST(Issue, 34) {
  struct S {
    S(const S &) = default;
    S(S &&) = default;
    S &operator=(const S &) = default;
    S &operator=(S &&) = default;

    Cexa::Experimental::variant<std::map<test_util::DeviceString, S>> value;
  };
}
#endif

// https://github.com/mpark/variant/pull/57
TEST(Issue, 57) {
  std::vector<Cexa::Experimental::variant<int, std::unique_ptr<int>>> vec;
  vec.emplace_back(0);
}

#endif

TEST_MAIN
