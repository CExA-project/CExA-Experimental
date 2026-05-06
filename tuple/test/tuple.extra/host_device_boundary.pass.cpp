// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <Kokkos_Core.hpp>

void test() {
  int i = 3;
  Kokkos::View<int*> v("view", 10);
  float f = 1.;
  cexa::tuple<int, Kokkos::View<int*>, float> t(i, v, f);

  Kokkos::parallel_for("loop", v.size(), KOKKOS_LAMBDA(int i) {
    CEXA_EXPECT_EQ(cexa::get<0>(t), 3);
    CEXA_EXPECT_EQ(cexa::get<2>(t), 1.f);
    cexa::get<1>(t)(i) = i;
  });

  auto h_v = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h_v, v);

  for (int i = 0; i < h_v.extent_int(0); i++) {
    CEXA_EXPECT_EQ(h_v(i), i);
  }
}

TEST(tuple_extra, host_device_boundary) {
  test();
}
