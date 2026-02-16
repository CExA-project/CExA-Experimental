// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
//
// This is a modified version of the tuple tests from llvm's libcxx tests,
// below is the original copyright statement
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03

// <tuple>

// template <class... Types> class tuple;

// template <class Alloc> tuple(allocator_arg_t, Alloc const&)

// See https://llvm.org/PR27684.

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

#if defined(CEXA_ON_DEVICE)
struct IncompleteType;
extern __device__ IncompleteType inc1;
extern __device__ IncompleteType inc2;
__device__ IncompleteType const& cinc1 = inc1;
__device__ IncompleteType const& cinc2 = inc2;
#else
struct IncompleteType;
extern IncompleteType inc1;
extern IncompleteType inc2;
IncompleteType const& cinc1 = inc1;
IncompleteType const& cinc2 = inc2;
#endif

// clang-format off
CEXA_TEST(tuple_cnstr, PR27684_contains_ref_to_incomplete_type, (
    using IT = IncompleteType;
    { // try calling tuple(Tp const&...)
        using Tup = cexa::tuple<const IT&, const IT&>;
        Tup t(cinc1, cinc2);
        CEXA_EXPECT_EQ(&cexa::get<0>(t), &inc1);
        CEXA_EXPECT_EQ(&cexa::get<1>(t), &inc2);
    }
    { // try calling tuple(Up&&...)
        using Tup = cexa::tuple<const IT&, const IT&>;
        // std::is_copy_constructible_v<const IT&>;
        // std::is_constructible_v<const IT&, const IT&>;
        Tup t(inc1, inc2);
        CEXA_EXPECT_EQ(&cexa::get<0>(t), &inc1);
        CEXA_EXPECT_EQ(&cexa::get<1>(t), &inc2);
    }
))
// clang-format on

#if defined(CEXA_ON_DEVICE)
struct IncompleteType {};
__device__ IncompleteType inc1;
__device__ IncompleteType inc2;
#else
struct IncompleteType {};
IncompleteType inc1;
IncompleteType inc2;
#endif
