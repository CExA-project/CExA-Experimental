//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DEFAULTONLY_H
#define DEFAULTONLY_H

#include <Kokkos_Macros.hpp>
#include "cexa_test_macros.hpp"

namespace {
#if defined(CEXA_ON_DEVICE)
__device__ inline int count = 0;
#else
inline int count = 0;
#endif
}

class DefaultOnly
{
    int data_;

    KOKKOS_INLINE_FUNCTION DefaultOnly(const DefaultOnly&);
    KOKKOS_INLINE_FUNCTION DefaultOnly& operator=(const DefaultOnly&);
public:

    KOKKOS_INLINE_FUNCTION DefaultOnly() : data_(-1) {++count;}
    KOKKOS_INLINE_FUNCTION ~DefaultOnly() {data_ = 0; --count;}

    KOKKOS_INLINE_FUNCTION friend bool operator==(const DefaultOnly& x, const DefaultOnly& y)
        {return x.data_ == y.data_;}
    KOKKOS_INLINE_FUNCTION friend bool operator< (const DefaultOnly& x, const DefaultOnly& y)
        {return x.data_ < y.data_;}
};

#endif // DEFAULTONLY_H
