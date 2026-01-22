#include <Kokkos_Variant.hpp>

#include <benchmark/benchmark.h>

#include <iostream>

#include <variant>

// Example of a code using virtual function, we want to parallelize its main
// loop using Kokkos
namespace virt {
struct A {
  __attribute__((noinline))virtual int f(int in) { return 4 + in; }
};
struct B : A {
__attribute__((noinline))  int f(int in) { return 6 + in; }
};
struct C : A {
__attribute__((noinline))  int f(int in) { return 2 * A::f(in); }
};

}  // namespace virt


static void Virt(benchmark::State& state) {
  const int size = 1000002;
  //std::vector<virt::A*> v(size);
  //for (int i = 0; i < size; i+=3) {
  //  v[0+i] = new virt::A;
  //  v[1+i] = new virt::B;
  //  v[2+i] = new virt::C;
  //}
  std::vector<virt::A*> v(3);
  //for (int i = 0; i < size; i+=3) {
    v[0] = new virt::A;
    v[1] = new virt::B;
    v[2] = new virt::C;
  //}

  std::vector<int> res(size);

  for (auto _ : state) {
    for (int i=0; i<size; ++i) {
      res[i] = v[i%3]->f(i);
    }
    benchmark::DoNotOptimize(res);
    //benchmark::ClobberMemory();
  }

  for (int i = 0; i < 3; i+=3) {
    delete v[i];
  }
}

namespace virtDevice {
struct A {
__attribute__((noinline))  KOKKOS_FUNCTION virtual int f(int in) { return 4 + in; }
};
struct B : A {
__attribute__((noinline))  KOKKOS_FUNCTION int f(int in) { return 6 + in; }
};
struct C : A {
__attribute__((noinline))  KOKKOS_FUNCTION int f(int in) { return 2 * A::f(in); }
};

}  // namespace virt


static void VirtDevice(benchmark::State& state) {
  const int size = 1000002;
  //std::vector<virt::A*> v(size);
  //Kokkos::View<virtDevice::A*, Kokkos::DefaultExecutionSpace> v("view", size);
  //for (int i = 0; i < size; i+=3) {
  //  v[0+i] = new virt::A;
  //  v[1+i] = new virt::B;
  //  v[2+i] = new virt::C;
  //}

  size_t deriv_size = std::max(sizeof(virtDevice::A), std::max(sizeof(virtDevice::B), sizeof(virtDevice::C)));

  // create
  void* deviceInstanceMemory = Kokkos::kokkos_malloc(deriv_size * size); // allocate memory on device
  Kokkos::parallel_for("initialize", size, KOKKOS_LAMBDA (const int i) {
      size_t virt_idx = deriv_size * i;

      // initialize on device
      switch (i%3) {
        case 0: 
          new (static_cast<virtDevice::A*>((void*)((uint8_t*)(deviceInstanceMemory) + virt_idx))) virtDevice::A();
          break;
        case 1: 
          new (static_cast<virtDevice::B*>((void*)((uint8_t*)(deviceInstanceMemory) + virt_idx))) virtDevice::B();
          break;
        case 2: 
          new (static_cast<virtDevice::C*>((void*)((uint8_t*)(deviceInstanceMemory) + virt_idx))) virtDevice::C();
          break;
      }

      });

  Kokkos::View<int*, Kokkos::DefaultExecutionSpace> res("res", size);

  for (auto _ : state) {
    Kokkos::parallel_for("loop", 
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, size),
        KOKKOS_LAMBDA(int i) {
          size_t virt_idx = deriv_size * i;
          virtDevice::A* a = static_cast<virtDevice::A*>((void*)((uint8_t*)(deviceInstanceMemory) + virt_idx));
          res[i] = a->f(i);
        });
    Kokkos::fence();
  }

  // cleanup
  Kokkos::kokkos_free(deviceInstanceMemory); // free
}

namespace FunctionPtr {
__attribute__((noinline))  int fA(int in) { return 4 + in; }
__attribute__((noinline))  int fB(int in) { return 6 + in; }
__attribute__((noinline))  int fC(int in) { return 2 * fA(in); }
}  // namespace FunctionPtr

static void FctPtr(benchmark::State& state) {
  const int size = 1000002;
  //std::vector<int (*)(int)> v(size);
  std::vector<int (*)(int)> v(3);

  //for (int i = 0; i < size; i+=3) {
    v[0] = &FunctionPtr::fA;
    v[1] = &FunctionPtr::fB;
    v[2] = &FunctionPtr::fC;
  //}

  std::vector<int> res(size);

  for (auto _ : state) {
    for (int i=0; i<size; ++i) {
      res[i] = v[i%3](i);
    }
    benchmark::DoNotOptimize(res);
    benchmark::ClobberMemory();
  }
}

namespace FunctionPtrDevice {
__attribute__((noinline))  KOKKOS_FUNCTION int fA(int in) { return 4 + in; }
__attribute__((noinline))  KOKKOS_FUNCTION int fB(int in) { return 6 + in; }
__attribute__((noinline))  KOKKOS_FUNCTION int fC(int in) { return 2 * fA(in); }
}  // namespace FunctionPtr

static void FctPtrDevice(benchmark::State& state) {
  const int size = 1000002;

  using fct_ptr = int (*)(int);

  Kokkos::View<int*, Kokkos::DefaultExecutionSpace> res("res", size);

  for (auto _ : state) {
    Kokkos::parallel_for("loop", 
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, size),
        KOKKOS_LAMBDA(int i) {
          fct_ptr f;
              if (i%3 == 0) {
              f = &FunctionPtrDevice::fA;
              } else if (i%3 == 1){
              f = &FunctionPtrDevice::fB;
              } else {
              f = &FunctionPtrDevice::fC;
              }
          res(i) = f(i);
        });
    Kokkos::fence();
  }
}

namespace Kokkos_port {
struct A {
  // `virtual` keyword has to be removed for portability with SYCL
__attribute__((noinline))  KOKKOS_FUNCTION /*virtual*/ int f(int in) { return 4 + in; }
};
struct B : A {
__attribute__((noinline))  KOKKOS_FUNCTION int f(int in) { return 6 + in; }
};
struct C : A {
__attribute__((noinline))  KOKKOS_FUNCTION int f(int in) { return 2 * A::f(in); }
};

using A_Hierarchy = cexa::experimental::variant<A, B, C>;

// Helper function that call the correct overload without vtable
KOKKOS_FUNCTION int call_f(const A_Hierarchy& var, int i) {
  return cexa::experimental::visit(
      [i](auto v) { return v.f(i); },
      var);
}
}  // namespace Kokkos_port

template <class MemorySpace>
static void Variants(benchmark::State& state) {
  const int size = 1000002;
  Kokkos::View<Kokkos_port::A_Hierarchy*, MemorySpace> v("view", size);
  auto v_h = Kokkos::create_mirror_view(v);
  for (int i = 0; i < size; i+=3) {
    v_h(i + 0) = Kokkos_port::A{};
    v_h(i + 1) = Kokkos_port::B{};
    v_h(i + 2) = Kokkos_port::C{};
  }

  Kokkos::deep_copy(v, v_h);

  Kokkos::View<int*, MemorySpace> res("res", size);

  for (auto _ : state) {
    Kokkos::parallel_for("loop", 
        Kokkos::RangePolicy<MemorySpace>(0, size),
        KOKKOS_LAMBDA(int i) {
          res(i) = Kokkos_port::call_f(v(i), i);
        });
    Kokkos::fence();
  }
}

__attribute__((noinline)) KOKKOS_FUNCTION int fA(int in) { return 4 + in; }
__attribute__((noinline)) KOKKOS_FUNCTION int fB(int in) { return 6 + in; }
__attribute__((noinline)) KOKKOS_FUNCTION int fC(int in) { return 2 * fA(in); }

template <class MemorySpace>
static void Switch(benchmark::State& state) {
  const int size = 1000002;

  Kokkos::View<int*, MemorySpace> res("res", size);

  for (auto _ : state) {
    Kokkos::parallel_for("loop", 
        Kokkos::RangePolicy<MemorySpace>(0, size),
        KOKKOS_LAMBDA(int i) {
          switch (i%3) {
            case 0: 
              res(i) = fA(i);
              break;
            case 1: 
              res(i) = fB(i);
              break;
            case 2: 
              res(i) = fC(i);
              break;
          }
        }
    );
    Kokkos::fence();
  }
}

static void SwitchNoKokkos(benchmark::State& state) {
  const int size = 1000002;

  std::vector<int> res(size);

  for (auto _ : state) {
    for (int i=0; i < size; ++i) {
        switch (i%3) {
          case 0: 
            res[i] = fA(i);
            break;
          case 1: 
            res[i] = fB(i);
            break;
          case 2: 
            res[i] = fC(i);
            break;
        }
    }
    benchmark::DoNotOptimize(res);
    //benchmark::ClobberMemory();
  }
}

static void VariantsNoKokkos(benchmark::State& state) {
  const int size = 1000002;
  std::vector<Kokkos_port::A_Hierarchy> v(size);
  for (int i = 0; i < size; i+=3) {
    v[i + 0] = Kokkos_port::A{};
    v[i + 1] = Kokkos_port::B{};
    v[i + 2] = Kokkos_port::C{};
  }

  std::vector<int> res(size);

  for (auto _ : state) {
    for (int i = 0; i<size; ++i) {
      res[i] = Kokkos_port::call_f(v[i], i);
    }
    benchmark::DoNotOptimize(res);
    benchmark::ClobberMemory();
  }
}

using stdA_Hierarchy = std::variant<Kokkos_port::A, Kokkos_port::B, Kokkos_port::C>;

// Helper function that call the correct overload without vtable
int call_f(const stdA_Hierarchy& var, int i) {
  return std::visit(
      [i](auto v) { return v.f(i); },
      var);
}

static void stdvariant(benchmark::State& state) {
  const int size = 1000002;
  Kokkos::View<stdA_Hierarchy*, Kokkos::DefaultHostExecutionSpace> v("view", size);
  auto v_h = Kokkos::create_mirror_view(v);
  for (int i = 0; i < size; i+=3) {
    v_h(i + 0) = Kokkos_port::A{};
    v_h(i + 1) = Kokkos_port::B{};
    v_h(i + 2) = Kokkos_port::C{};
  }

  Kokkos::deep_copy(v, v_h);

  Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace> res("res", size);

  for (auto _ : state) {
    for (int i = 0; i<size; ++i) {
          res(i) = call_f(v(i), i);
    }
    benchmark::DoNotOptimize(res);
    benchmark::ClobberMemory();
  }
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  srand(0);

  benchmark::RegisterBenchmark("virt", Virt);
  benchmark::RegisterBenchmark("fctPtr", FctPtr);
  benchmark::RegisterBenchmark("variants_host", Variants<Kokkos::DefaultHostExecutionSpace>);
  benchmark::RegisterBenchmark("variantsNoKokkos", VariantsNoKokkos);
  benchmark::RegisterBenchmark("switch_host", Switch<Kokkos::DefaultHostExecutionSpace>);
  benchmark::RegisterBenchmark("switch_hostnoKokkos", SwitchNoKokkos);
  benchmark::RegisterBenchmark("stdvariant(host)", stdvariant);
  benchmark::RegisterBenchmark("VirtDevice", VirtDevice);
  benchmark::RegisterBenchmark("fctPtrDevice", FctPtrDevice);
  benchmark::RegisterBenchmark("switch_device", Switch<Kokkos::DefaultExecutionSpace>);
  benchmark::RegisterBenchmark("variants_device", Variants<Kokkos::DefaultExecutionSpace>);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
