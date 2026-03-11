# ArchInfo

A small utility library for printing useful information about the system a
Kokkos program runs on.

## Build

The library can be built using CMake, it depends on Kokkos
```bash
git clone https://github.com/CExA-project/CExA-Experimental
cd CExA-Experimental/archInfo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=\<path/to/kokkos/install\>
cmake --build build
cmake --install
```

The tests can be enabled by adding the option `-DCEXA_ARCHINFO_ENABLE_TESTS=ON`
to CMake.

## Usage

The library can be included in a CMake project using `find_package`
```cmake
find_package(Kokkos REQUIRED)
find_package(CexaArchInfo 0.1.0 REQUIRED)

add_executable(main)
target_link_libraries(main PRIVATE cexa::archInfo Kokkos::kokkos)
```

### API

#### Information about the operating system

```cpp
Kokkos::print_os_info(std::cout);
```

Possible output on a Linux system:
```text
OS Type   : Linux
   Name   : Red Hat Enterprise Linux 9.6 (Plow)
   Kernel : 5.14.0-570.69.1.el9_6.x86_64
```

#### Information about the CPU

```cpp
Kokkos::print_host_info(std::cout);
```

Possible output:
```text
CPU Model   : AMD EPYC 9654 96-Core Processor
    Cores   : 96
    Threads : 384
    Sockets : 2
```

#### Information about the GPU

```cpp
Kokkos::print_device_info(std::cout);
```

Possible output:
```text
GPU Model           : AMD Instinct MI300A
    Arch            : gfx942:sramecc+:xnack-
    Runtime Version : 6.3.42134
    Driver Version  : 6.3.42134
```
