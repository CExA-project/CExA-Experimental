### memInfo
This is a wrapper for the device _memGetInfo_ function, but it's also available for unified memory space and host space.

### Usage 
```
Kokkos::Experimental::memInfo<MemorySpace>(&free, &total);
```
- `total`: Amount of RAM on the system (HBM/DRAM)
- `free`: Available memory