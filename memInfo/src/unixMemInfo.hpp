#ifndef KOKKOS_UNIX_MEMINFO_HPP
#define KOKKOS_UNIX_MEMINFO_HPP

#include <unistd.h>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

#include <Kokkos_Core.hpp>

namespace Kokkos::Experimental {

namespace {
constexpr size_t NO_LIMIT = 1ULL << 50ULL;  // No limit (like PAGE_COUNT_MAX)
constexpr int OVERCOMMIT_DISABLED = 2;
// Memory info keys
constexpr char MEM_FREE_KEY[]     = "MemFree:";
constexpr char MEM_TOTAL_KEY[]    = "MemTotal:";
constexpr char COMMITTED_AS_KEY[] = "Committed_AS:";
constexpr char COMMIT_LIMIT_KEY[] = "CommitLimit:";
// Cgroup v1 memory info
constexpr char CGROUP_PROCS[]    = "cgroup.procs";
constexpr char MEM_LIMIT_BYTES[] = "memory.limit_in_bytes";
constexpr char MEM_USAGE_BYTES[] = "memory.usage_in_bytes";
// Paths
constexpr char MEMINFO_PATH[]    = "/proc/meminfo";
constexpr char OVERCOMMIT_PATH[] = "/proc/sys/vm/overcommit_memory";
constexpr char CGROUP_MEM_PATH[] = "/sys/fs/cgroup/memory";
constexpr char CGROUP_V2_PATH[]  = "/sys/fs/cgroup/cgroup.controllers";
}  // namespace

template <typename Space>
void MemGetInfo(size_t* free, size_t* total);

// On some systems, overcommit is disabled, and the kernel does not allow
// memory allocation beyond the commit limit. This means that allocations
// that touch only a small amount of memory are still counted at their full
// size. man proc_sys_vm
inline bool is_overcommit_disabled() {
  std::ifstream overcommit_file(OVERCOMMIT_PATH);
  int overcommit_value = 0;

  if (overcommit_file.is_open()) {
    overcommit_file >> overcommit_value;
    overcommit_value = (overcommit_file.fail()) ? 0 : overcommit_value;
    return (overcommit_value == OVERCOMMIT_DISABLED);
  }
  return false;
}

// Extract a value from /proc/meminfo
inline size_t get_meminfo_value(const char* key) {
  std::ifstream meminfo(MEMINFO_PATH);
  size_t value = 0;
  std::string line;

  if (meminfo.is_open()) {
    while (std::getline(meminfo, line)) {
      if (line.find(key) != std::string::npos) {
        std::istringstream iss(line);
        iss.ignore(256, ':');
        iss >> value;
        value = (iss.fail()) ? 0 : value * 1024;
        break;
      }
    }
  }
  return value;
}

// Extract a value from a cgroup file
inline size_t get_cgroup_value(const char* path) {
  std::ifstream cgroup_file(path);
  size_t value = 0;

  if (cgroup_file.is_open()) {
    cgroup_file >> value;
    value = (cgroup_file.fail()) ? 0 : value;
  }
  return value;
}

// Check if a process is in the cgroup.procs file
inline bool is_pid_in_cgroup_procs(const char* cgroup_procs_path,
                                   const pid_t pid) {
  std::ifstream cgroup_procs(cgroup_procs_path);
  pid_t proc_id = 0;

  if (cgroup_procs.is_open()) {
    while (cgroup_procs >> proc_id) {
      if (proc_id == pid) {
        return true;
      }
    }
  }
  return false;
}

// Find out if memory controller is enabled (cgroup v1)
// Check in /proc/<pid>/cgroup
inline bool is_cgroup_mem_control_enabled() {
  const pid_t pid = getpid();
  std::ifstream cgroup_file("/proc/" + std::to_string(pid) + "/cgroup");
  std::string line;

  if (cgroup_file.is_open()) {
    while (std::getline(cgroup_file, line)) {
      if (line.find("memory") != std::string::npos) {
        return true;
      }
    }
  }
  return false;
}

inline bool using_cgroup_v2() {
  std::ifstream cgroup_file(CGROUP_V2_PATH);
  return cgroup_file.is_open();
}

// Find the cgroup memory path for the current process
// Verify if the process is in the cgroup.procs file
inline std::string find_cgroup_memory_path() {
  const pid_t pid = getpid();
  std::ifstream cgroup_file("/proc/" + std::to_string(pid) + "/cgroup");
  std::string cgroup_path;

  if (!using_cgroup_v2() && cgroup_file.is_open()) {
    std::string line;
    while (std::getline(cgroup_file, line)) {
      if (line.find(":memory:") != std::string::npos) {
        const size_t pos = line.find_last_of(':');
        if (pos != std::string::npos) {
          cgroup_path = line.substr(pos + 1);
          if (is_pid_in_cgroup_procs(
                  (CGROUP_MEM_PATH + cgroup_path + "/" + CGROUP_PROCS).c_str(),
                  pid)) {
            return CGROUP_MEM_PATH + cgroup_path;
          }
          const size_t last_slash = cgroup_path.find_last_of('/');
          if (last_slash != std::string::npos) {
            std::string parent_path = cgroup_path.substr(0, last_slash);
            if (is_pid_in_cgroup_procs(
                    (CGROUP_MEM_PATH + parent_path + "/" + CGROUP_PROCS)
                        .c_str(),
                    pid)) {
              return CGROUP_MEM_PATH + parent_path;
            }
          }
        }
      }
    }
  }

  // Fallback to the default path
  return CGROUP_MEM_PATH + cgroup_path;
}

// Single node memory info
template <>
inline void MemGetInfo<Kokkos::HostSpace>(size_t* free, size_t* total) {
  static const bool overcommit_disabled = is_overcommit_disabled();
  static const bool cgroup_mem_enabled  = is_cgroup_mem_control_enabled();
  static const std::string cgroup_mem_path =
      cgroup_mem_enabled ? find_cgroup_memory_path() : std::string{};

  // Cgroup memory info
  if (cgroup_mem_enabled) {
    const size_t mem_limit =
        get_cgroup_value((cgroup_mem_path + "/" + MEM_LIMIT_BYTES).c_str());
    const size_t mem_usage =
        get_cgroup_value((cgroup_mem_path + "/" + MEM_USAGE_BYTES).c_str());

    if (mem_limit == 0 || mem_limit > NO_LIMIT) {
      if (overcommit_disabled) {
        *total            = get_meminfo_value(COMMIT_LIMIT_KEY);
        const size_t used = get_meminfo_value(COMMITTED_AS_KEY);
        *free             = (*total > used) ? *total - used : 0;
      } else {
        *total = get_meminfo_value(MEM_TOTAL_KEY);
        *free  = get_meminfo_value(MEM_FREE_KEY);
      }
    } else {
      *total = mem_limit;
      *free  = (mem_limit > mem_usage) ? mem_limit - mem_usage : 0;
    }
    return;
  }

  // System memory info
  if (overcommit_disabled) {
    *total            = get_meminfo_value(COMMIT_LIMIT_KEY);
    const size_t used = get_meminfo_value(COMMITTED_AS_KEY);
    *free             = (*total > used) ? *total - used : 0;
  } else {
    *total = get_meminfo_value(MEM_TOTAL_KEY);
    *free  = get_meminfo_value(MEM_FREE_KEY);
  }
}

}  // namespace Kokkos::Experimental

#endif  // KOKKOS_UNIX_MEMINFO_HPP
