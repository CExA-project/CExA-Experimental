# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
add_tuple_test(eq.pass.cpp)
add_tuple_test(lt.pass.cpp)
add_tuple_compile_test(size_incompatible_three_way.compile.pass.cpp)
add_tuple_test(three_way.pass.cpp)
