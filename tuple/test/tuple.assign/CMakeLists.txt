# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
add_tuple_test(const_convert_copy.pass.cpp)
add_tuple_test(const_convert_move.pass.cpp)
add_tuple_test(const_copy.pass.cpp)
add_tuple_test(const_move.pass.cpp)
add_tuple_test(const_pair.pass.cpp assign) # FIXME: host only
add_tuple_test(const_pair_copy.pass.cpp) # FIXME: host only
add_tuple_test(const_pair_move.pass.cpp) # FIXME: host only
add_tuple_test(convert_copy.pass.cpp assign)
add_tuple_test(convert_move.pass.cpp assign)
add_tuple_test(copy.pass.cpp assign)
add_tuple_test(derived_from_tuple_like.pass.cpp)
add_tuple_compile_test(laziness.pass.cpp)
add_tuple_test(move.pass.cpp assign)
add_tuple_test(move_pair.pass.cpp assign) # FIXME: host only
