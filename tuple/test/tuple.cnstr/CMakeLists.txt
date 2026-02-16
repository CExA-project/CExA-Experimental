# SPDX-FileCopyrightText: 2026 CExA-project
# SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
add_tuple_test(default.pass.cpp)
add_tuple_test(dtor.pass.cpp)
add_tuple_test(UTypes.pass.cpp)
add_tuple_test(const_Types.pass.cpp)
add_tuple_test(const_move_pair.pass.cpp)
add_tuple_test(const_pair.pass.cpp)
add_tuple_test(convert_const_move.pass.cpp)
add_tuple_test(convert_copy.pass.cpp)
add_tuple_test(convert_move.pass.cpp)
add_tuple_test(convert_non_const_copy.pass.cpp)
add_tuple_test(copy.pass.cpp)
add_tuple_test(deduct.pass.cpp)
add_tuple_test(empty_tuple_trivial.compile.pass.cpp) # doesn't need a host and a device version
add_tuple_test(move.pass.cpp)
add_tuple_test(move_pair.pass.cpp)
add_tuple_test(non_const_pair.pass.cpp)
add_tuple_test(recursion_depth.pass.cpp)
add_tuple_test(test_lazy_sfinae.pass.cpp)
add_tuple_compile_test(PR20855_tuple_ref_binding_diagnostics.pass.cpp) # only has compile checks for the moment
add_tuple_test(PR22806_constrain_tuple_like_ctor.pass.cpp)
# add_tuple_test(PR23256_constrain_UTypes_ctor.pass.cpp) # FIXME:
add_tuple_test(PR27684_contains_ref_to_incomplete_type.pass.cpp)
add_tuple_test(PR31384.pass.cpp)

# alloc.pass.cpp
# alloc_UTypes.pass.cpp
# alloc_const_Types.pass.cpp
# alloc_const_move_pair.pass.cpp
# alloc_const_pair.pass.cpp
# alloc_convert_const_move.pass.cpp
# alloc_convert_copy.pass.cpp
# alloc_convert_move.pass.cpp
# alloc_convert_non_const_copy.pass.cpp
# alloc_copy.pass.cpp
# alloc_move.pass.cpp
# alloc_move_pair.pass.cpp
# alloc_non_const_pair.pass.cpp
