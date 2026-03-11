#ifndef CEXA_MACOS_ARCHINFO_HPP
#define CEXA_MACOS_ARCHINFO_HPP

#include "cexa_ArchInfo.hpp"

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <sys/sysctl.h>

// This header is only included in a single cpp file, and adding inline to the
// functions leads to undefined references with apple clang
// NOLINTBEGIN(misc-definitions-in-headers)
namespace cexa::experimental {

std::optional<std::string> get_sysctl_string(std::string_view name) {
  std::size_t size = 0;

  sysctlbyname(name.data(), nullptr, &size, nullptr, 0);
  std::string value(size, '\0');

  if (sysctlbyname(name.data(), value.data(), &size, nullptr, 0) != 0) {
    return std::nullopt;
  }

  if (value.size() > 0 && value.back() == '\0') {
    value.pop_back();
  }

  return value;
}

std::optional<std::int64_t> get_sysctl_int(std::string_view name) {
  std::size_t size = 0;
  sysctlbyname(name.data(), nullptr, &size, nullptr, 0);

  if (size == 2) {
    std::int16_t value;
    if (sysctlbyname(name.data(), &value, &size, nullptr, 0) != 0) {
      return std::nullopt;
    }
    return value;
  } else if (size == 4) {
    std::int32_t value;
    if (sysctlbyname(name.data(), &value, &size, nullptr, 0) != 0) {
      return std::nullopt;
    }
    return value;
  } else if (size == 8) {
    std::int64_t value;
    if (sysctlbyname(name.data(), &value, &size, nullptr, 0) != 0) {
      return std::nullopt;
    }
    return value;
  } else {
    return std::nullopt;
  }
}

std::optional<std::string> extract_plist_value(const std::string& line) {
  if (line.empty()) {
    return std::nullopt;
  }

  size_t pos = line.find("<string>");
  if (pos == std::string::npos) {
    return std::nullopt;
  };
  pos += 8;

  size_t end = line.find("</string>", pos);
  if (end == std::string::npos || pos == end) {
    return std::nullopt;
  };

  return line.substr(pos, end - pos);
}

size_t get_physical_socket_count() {
  return get_sysctl_int("hw.packages").value_or(-1);
}

size_t get_core_count_per_socket() {
  return get_sysctl_int("machdep.cpu.cores_per_package").value_or(-1);
}

size_t get_thread_count_per_socket() {
  return get_sysctl_int("machdep.cpu.logical_per_package").value_or(-1);
}

std::string get_cpu_model_name() {
  return get_sysctl_string("machdep.cpu.brand_string").value_or("ERROR");
}

std::string get_sys_name() {
  std::ifstream plist_file("/System/Library/CoreServices/SystemVersion.plist");

  if (!plist_file.is_open()) return "macOS";

  std::string name, version;
  std::string line;
  while (std::getline(plist_file, line)) {
    if (line.find("<key>ProductName</key>") != std::string::npos) {
      std::getline(plist_file, line);
      name = extract_plist_value(line).value_or("ERROR");
    } else if (line.find("<key>ProductVersion</key>") != std::string::npos) {
      std::getline(plist_file, line);
      version = extract_plist_value(line).value_or("");
    }
    line.clear();
  }

  return version.empty() ? name : name + " " + version;
}

std::string get_sys_type() {
  return get_sysctl_string("kern.ostype").value_or("ERROR");
}

std::string get_kernel_version() {
  return get_sysctl_string("kern.osrelease").value_or("ERROR");
}

}  // namespace cexa::experimental
// NOLINTEND(misc-definitions-in-headers)

#endif
