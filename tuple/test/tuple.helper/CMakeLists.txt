# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
add_tuple_compile_test(tuple.include.array.pass.cpp)
add_tuple_compile_test(tuple.include.ranges.pass.cpp)
add_tuple_compile_test(tuple.include.utility.pass.cpp)
add_tuple_compile_test(tuple_element.pass.cpp)
add_tuple_compile_test(tuple_size.pass.cpp)
add_tuple_compile_test(tuple_size_incomplete.pass.cpp)
add_tuple_test(tuple_size_structured_bindings.pass.cpp)
add_tuple_compile_test(tuple_size_v.pass.cpp)
add_tuple_compile_test(tuple_size_value_sfinae.pass.cpp)
