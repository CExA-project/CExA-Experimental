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

//  Tuples of smart pointers; based on bug #18350
//  auto_ptr doesn't have a copy constructor that takes a const &, but tuple does.

#include <memory>

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>

TEST(tuple_general, tuple_smartptr) {
    {
    cexa::tuple<std::unique_ptr<char>> up;
    cexa::tuple<std::shared_ptr<char>> sp;
    cexa::tuple<std::weak_ptr  <char>> wp;
    }
    {
    cexa::tuple<std::unique_ptr<char[]>> up;
    cexa::tuple<std::shared_ptr<char[]>> sp;
    cexa::tuple<std::weak_ptr  <char[]>> wp;
    }
    // Smart pointers of type 'T[N]' are not tested here since they are not
    // supported by the standard nor by libc++'s implementation.
    // See https://reviews.llvm.org/D21320 for more information.
}
