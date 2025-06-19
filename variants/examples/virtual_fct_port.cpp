// SPDX-FileCopyrightText: 2025 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
//
// Example of porting code using virtual functions to Kokkos by using variants

#include <Kokkos_Core.hpp>
#include <Kokkos_Variant.hpp>

#include <iostream>

// Example of a code using virtual function, we want to parallelize its main
// loop using Kokkos
namespace original {
struct A {
  virtual int f(int in) { return 4 + in; }
};
struct B : A {
  int f(int in) { return 6 + in; }
};
struct C : A {
  int f(int in) { return 2 * A::f(in); }
};

void do_computation() {
  std::vector<A*> v(3);
  v[0] = new A;
  v[1] = new B;
  v[2] = new C;

  for (int i = 0; i < 3; ++i) {
    // This won't work on GPU (see here
    // https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Kokkos-and-Virtual-Functions.html)
    std::cout << i << ": " << v[i]->f(i) << "\n";
  }

  for (int i = 0; i < 3; ++i) {
    delete v[i];
  }
}
}  // namespace original

// Showing how the code could be ported with minimal changes using
// Kokkos::variant
namespace Kokkos_port {
struct A {
  // The `virtual` keyword has to be removed for the code to compile on SYCL
  KOKKOS_FUNCTION /*virtual*/ int f(int in) { return 4 + in; }
};
struct B : A {
  KOKKOS_FUNCTION int f(int in) { return 6 + in; }
};
struct C : A {
  KOKKOS_FUNCTION int f(int in) { return 2 * A::f(in); }
};

using A_Hierarchy = cexa::experimental::variant<A, B, C>;

// Helper function that will directly call the correct overload without using
// any vtable
KOKKOS_FUNCTION int call_f(const A_Hierarchy& var, int i) {
  return cexa::experimental::visit([i](auto v) { return v.f(i); }, var);
}

void do_computation() {
  Kokkos::View<A_Hierarchy*> v("view", 3);
  auto mv = Kokkos::create_mirror_view(v);
  mv(0)   = A{};
  mv(1)   = B{};
  mv(2)   = C{};

  Kokkos::deep_copy(v, mv);
  Kokkos::parallel_for(
      Kokkos::RangePolicy(0, 3),
      KOKKOS_LAMBDA(int i) { Kokkos::printf("%i: %i\n", i, call_f(v(i), i)); });
}
}  // namespace Kokkos_port

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  std::cout << "Original code:\n";
  original::do_computation();

  std::cout << "Kokkos port:\n";
  // Output may be in a different order but the 3 values should be the same
  Kokkos_port::do_computation();
}
