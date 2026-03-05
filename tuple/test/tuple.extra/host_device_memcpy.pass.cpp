// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
//
// This is a modified version of the tuple tests from llvm's libcxx tests,
// below is the original copyright statement
#include <tuple.hpp>
#include <support/cexa_test_macros.hpp>
#include <support/test_macros.h>
#include <Kokkos_Core.hpp>

void test() {
  Kokkos::View<cexa::tuple<int, float, Kokkos::pair<int, int>>*> d_v("d_v", 10);

  Kokkos::parallel_for("loop", d_v.size(), KOKKOS_LAMBDA(int i) {
    cexa::get<0>(d_v(i)) = i;
    cexa::get<1>(d_v(i)) = 1.;
    Kokkos::pair<int, int>& p = cexa::get<2>(d_v(i));
    p.first = i / 2;
    p.second = i / 3;
  });

  auto h_v = Kokkos::create_mirror_view(d_v);
  Kokkos::deep_copy(h_v, d_v);

  for (int i = 0; i < h_v.extent_int(0); i++) {
    auto [i_, f, p] = h_v(i);
    CEXA_EXPECT_EQ(i_, i);
    CEXA_EXPECT_EQ(f, 1.f);
    CEXA_EXPECT_EQ(p, (Kokkos::pair{i / 2, i / 3}));
  }
}

TEST(tuple_extra, host_device_memcpy) {
  test();
}
