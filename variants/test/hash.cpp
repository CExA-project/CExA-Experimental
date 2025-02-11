// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#include <Kokkos_Variant.hpp>

#include <string>

#include <gtest/gtest.h>

#include "util.hpp"

// No need to test interaction between Kokkos::Variant and std::hash on GPU
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) &&             \
    !defined(KOKKOS_ENABLE_SYCL)
TEST(Hash, Monostate) {
  Cexa::Experimental::variant<int, Cexa::Experimental::monostate, std::string>
      v(Cexa::Experimental::monostate{});
  // Construct hash function objects.
  std::hash<Cexa::Experimental::monostate> monostate_hash;
  std::hash<Cexa::Experimental::variant<int, Cexa::Experimental::monostate,
                                        std::string>>
      variant_hash;
  // Check the hash.
  EXPECT_NE(monostate_hash(Cexa::Experimental::monostate{}), variant_hash(v));
}

TEST(Hash, String) {
  Cexa::Experimental::variant<int, std::string> v("hello");
  EXPECT_EQ("hello", Cexa::Experimental::get<std::string>(v));
  // Construct hash function objects.
  std::hash<std::string> string_hash;
  std::hash<Cexa::Experimental::variant<int, std::string>> variant_hash;
  // Check the hash.
  EXPECT_NE(string_hash("hello"), variant_hash(v));
}

#endif

TEST_MAIN
