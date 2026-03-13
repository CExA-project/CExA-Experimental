#ifndef CEXA_WINDWOS_ARCHINFO_HPP
#define CEXA_WINDWOS_ARCHINFO_HPP

#include "cexa_ArchInfo.hpp"

#include <bit>
#include <optional>
#include <string>
#include <array>
#include <string_view>
#include <vector>
#include <windows.h>
#include <intrin.h>

// This header is only included in a single cpp file
// NOLINTBEGIN(misc-definitions-in-headers)
namespace cexa::impl {

template <class T>
std::optional<T> read_registry_value(std::string_view path,
                                     std::string_view key) {
  HKEY hKey;
  if (ERROR_SUCCESS !=
      RegOpenKeyExA(HKEY_LOCAL_MACHINE, path.data(), 0, KEY_READ, &hKey)) {
    return std::nullopt;
  }

  T value;
  void* buffer      = &value;
  DWORD buffer_size = sizeof(value);

  if constexpr (std::is_same_v<T, std::string>) {
    if (ERROR_SUCCESS != RegGetValueA(hKey, nullptr, key.data(), RRF_RT_ANY,
                                      nullptr, nullptr, &buffer_size)) {
      RegCloseKey(hKey);
      return std::nullopt;
    }
    value.resize(buffer_size, '\0');
    buffer = value.data();
  }

  LSTATUS err = RegGetValueA(hKey, nullptr, key.data(), RRF_RT_ANY, nullptr,
                             buffer, &buffer_size);
  RegCloseKey(hKey);

  if (err != ERROR_SUCCESS) {
    return std::nullopt;
  }

  return value;
}

std::size_t get_physical_socket_count() {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorPackage, nullptr, &length);
  std::vector<std::byte> proc_info(length);
  if (!GetLogicalProcessorInformationEx(
          RelationProcessorPackage,
          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)proc_info.data(),
          &length)) {
    return -1;
  }

  std::size_t num_numa_node = 0;
  for (std::size_t i = 0; i < length;) {
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            proc_info.data() + i);

    if (info->Relationship == RelationProcessorPackage) {
      num_numa_node++;
    }

    i += info->Size;
  }
  return num_numa_node;
}

std::size_t get_core_count_per_socket() {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
  std::vector<std::byte> proc_info(length);
  if (!GetLogicalProcessorInformationEx(
          RelationProcessorCore,
          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)proc_info.data(),
          &length)) {
    return -1;
  }

  std::size_t num_cores = 0;
  for (std::size_t i = 0; i < length;) {
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            proc_info.data() + i);

    if (info->Relationship == RelationProcessorCore) {
      num_cores++;
    }

    i += info->Size;
  }

  return num_cores / get_physical_socket_count();
}

std::size_t get_thread_count_per_socket() {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorPackage, nullptr, &length);
  std::vector<std::byte> proc_info(length);
  if (!GetLogicalProcessorInformationEx(
          RelationProcessorPackage,
          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)proc_info.data(),
          &length)) {
    return -1;
  }

  for (std::size_t i = 0; i < length;) {
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            proc_info.data() + i);

    if (info->Relationship == RelationProcessorPackage) {
      std::size_t num_threads = 0;
      for (WORD group = 0; group < info->Processor.GroupCount; group++) {
        num_threads += std::popcount(info->Processor.GroupMask[group].Mask);
      }
      return num_threads;
    }

    i += info->Size;
  }
  return -1;
}

// FIXME: This will not work on arm
std::string get_cpu_model_name() {
  std::array<int, 4> regs;

  // The CPU brand can be obtained using ids 0x80000002 to 0x80000004, we check
  // that these ids are below the maximum id available.
  __cpuid(regs.data(), 0x80000000);
  int max_ext_id = regs[0];

  if (static_cast<unsigned int>(max_ext_id) < 0x80000004) {
    return "N/A";
  }

  constexpr std::size_t regs_size = 16;
  char name[3 * regs_size + 1]    = {};

  __cpuid(regs.data(), 0x80000002);
  memcpy(name, regs.data(), regs_size);
  __cpuid(regs.data(), 0x80000003);
  memcpy(name + regs_size, regs.data(), regs_size);
  __cpuid(regs.data(), 0x80000004);
  memcpy(name + 2 * regs_size, regs.data(), regs_size);

  return std::string(name);
}

std::string get_sys_name() {
  std::optional<std::string> sys_name = read_registry_value<std::string>(
      "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "ProductName");

  return sys_name.value_or("ERROR");
}

std::string get_sys_type() { return "Windows"; }

std::string get_kernel_version() {
  std::optional<std::string> current_build = read_registry_value<std::string>(
      "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "CurrentBuildNumber");
  if (!current_build) {
    return "ERROR";
  }

  std::optional<DWORD> ubr = read_registry_value<DWORD>(
      "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "UBR");
  if (!ubr) {
    return "ERROR";
  }

  return current_build.value() + "." + std::to_string(ubr.value());
}

}  // namespace cexa::impl
// NOLINTEND(misc-definitions-in-headers)

#endif
