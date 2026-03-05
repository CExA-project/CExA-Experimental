## ArchInfo

## Usage

### Information about the operating system:

```cpp
Kokkos::print_os(std::cout);
```

Possible output on a Linux system:
```text
OS Type   : Linux
   Name   : Red Hat Enterprise Linux 9.6 (Plow)
   Kernel : 5.14.0-570.69.1.el9_6.x86_64
```

### Information about the CPU:

```cpp
Kokkos::print_cpu(std::cout);
```

Possible output:
```text
CPU Model   : AMD EPYC 9654 96-Core Processor
    Cores   : 96
    Threads : 384
    Sockets : 2
```

### Information about the GPU:

```cpp
Kokkos::print_gpu(std::cout);
```

Possible output:
```text
GPU Model           :AMD Instinct MI300A
    Arch            :gfx942:sramecc+:xnack-
    Runtime Version :6.3.42134
    Driver Version  :6.3.42134
```
