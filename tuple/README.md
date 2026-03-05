<!--
SPDX-FileCopyrightText: 2026 CExA-project

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Tuple

This folder contains an implementation of a `tuple` type, following the `std::tuple` interface, suitable for use in Kokkos programs. The types and functions are contained in the `cexa` namespace

## Available features

Alongside the `cexa::tuple` type, the following functions are made available:
- `cexa::make_tuple`
- `cexa::tie`
- `cexa::forward_as_tuple`
- `cexa::tuple_cat`
- `cexa::get(cexa::tuple)`
- `cexa::swap(cexa::tuple, cexa::tuple)`

The following helper classes are available:
- `cexa::tuple_size<cexa::tuple>`
- `cexa::tuple_element<std::size_t, cexa::tuple>`
- `cexa::ignore`

Specializations of the following helper classes from the `std` namespace are provided:
- `std::tuple_size<cexa::tuple>`
- `std::tuple_element<std::size_t, cexa::tuple>`
- `std::basic_common_reference`
- `std::common_type`

`cexa::tuple` implements the tuple protocol, which means that it can be used with structured bindings:
```cpp
cexa::tuple<int, float, char> tup(3, 4.5, 'c');
auto [i, f, c] = tup;
```

## interoperability with std::tuple

`cexa::tuple` is currently not interoperable with `std::tuple`
- A `cexa::tuple` cannot be constructed from a `std::tuple`
- The `cexa::` functions will not work with a `std::tuple` argument

## Missing features

- `std::allocator_arg_t` constructors
- compatibility with `std::complex` (which is tuple-like from C++26 onwards)
