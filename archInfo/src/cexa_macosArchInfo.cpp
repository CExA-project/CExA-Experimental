// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

#if defined(__APPLE__)

#include "cexa_ArchInfo.hpp"

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <sys/sysctl.h>

namespace cexa::impl {

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
  }

  if (size == 4) {
    std::int32_t value;
    if (sysctlbyname(name.data(), &value, &size, nullptr, 0) != 0) {
      return std::nullopt;
    }
    return value;
  }

  if (size == 8) {
    std::int64_t value;
    if (sysctlbyname(name.data(), &value, &size, nullptr, 0) != 0) {
      return std::nullopt;
    }
    return value;
  }

  return std::nullopt;
}

// Extracts the value from an XML string node (<string>value</string>)
std::optional<std::string> extract_plist_value(const std::string& line) {
  if (line.empty()) {
    return std::nullopt;
  }

  std::size_t pos = line.find("<string>");
  if (pos == std::string::npos) {
    return std::nullopt;
  };
  pos += 8;

  std::size_t end = line.find("</string>", pos);
  if (end == std::string::npos || pos == end) {
    return std::nullopt;
  };

  return line.substr(pos, end - pos);
}

}  // namespace cexa::impl

namespace cexa {

std::size_t get_physical_socket_count() {
  return impl::get_sysctl_int("hw.packages").value_or(-1);
}

std::size_t get_core_count_per_socket() {
  return impl::get_sysctl_int("machdep.cpu.cores_per_package").value_or(-1);
}

std::size_t get_thread_count_per_socket() {
  return impl::get_sysctl_int("machdep.cpu.logical_per_package").value_or(-1);
}

std::string get_cpu_model_name() {
  return impl::get_sysctl_string("machdep.cpu.brand_string").value_or("ERROR");
}

std::string get_sys_name() {
  // We have to read the values from this XML-like file. The file contains a
  // sequence of key and string XML nodes
  std::ifstream plist_file("/System/Library/CoreServices/SystemVersion.plist");

  if (!plist_file.is_open()) {
    return "ERROR";
  }

  std::string name, version;
  std::string line;
  while (std::getline(plist_file, line)) {
    if (line.find("<key>ProductName</key>") != std::string::npos) {
      std::getline(plist_file, line);
      name = impl::extract_plist_value(line).value_or("ERROR");
    } else if (line.find("<key>ProductVersion</key>") != std::string::npos) {
      std::getline(plist_file, line);
      version = impl::extract_plist_value(line).value_or("");
    }
    line.clear();
  }

  return version.empty() ? name : name + " " + version;
}

std::string get_sys_type() {
  return impl::get_sysctl_string("kern.ostype").value_or("ERROR");
}

std::string get_kernel_version() {
  return impl::get_sysctl_string("kern.osrelease").value_or("ERROR");
}

}  // namespace cexa

#endif
