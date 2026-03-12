#ifndef CEXA_UNIX_ARCHINFO_HPP
#define CEXA_UNIX_ARCHINFO_HPP

#include "cexa_ArchInfo.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_set>

// This header is only included in a single cpp file
// NOLINTBEGIN(misc-definitions-in-headers)
namespace cexa::experimental {

// Extract a value from /proc/sys/ files
std::optional<std::string> get_proc_sys_value(const char* files) {
  std::ifstream proc_sys_file("/proc/sys/" + std::string(files));

  if (!proc_sys_file.is_open()) {
    return std::nullopt;
  }

  std::string value;
  std::getline(proc_sys_file, value);
  if (value.empty()) {
    return std::nullopt;
  }

  return value;
}

struct cpu_topology {
  std::size_t n_sockets          = -1;
  std::size_t procs_per_socket   = -1;
  std::size_t threads_per_socket = -1;
};

// Reads informations about the cpus from /sys/devices/system/cpu. This assumes
// that we have read access to /sys/
// NOTE: simpler alternatives would be:
// - reading the values from /proc/cpuinfo (doesn't work on arm)
// - use hwloc (requires hwloc to be installed on the system)
cpu_topology init_cpu_topology() {
  namespace fs = std::filesystem;

  std::unordered_set<std::string> package_ids, core_ids;
  int n_threads = 0;

  for (auto& entry : fs::directory_iterator("/sys/devices/system/cpu")) {
    if (!entry.is_directory()) {
      continue;
    }

    // we only want to iterate the cpu0, cpu1, ... directories
    std::string name = entry.path().filename().string();
    if (name.find("cpu") != 0 || name.size() < 4 || !std::isdigit(name[3])) {
      continue;
    }

    n_threads++;

    fs::path topology = entry.path() / "topology";

    // socket id
    std::ifstream package_id_file(topology / "physical_package_id");
    if (!package_id_file.is_open()) {
      return cpu_topology{};
    }

    std::string socket_id;
    package_id_file >> socket_id;
    package_ids.insert(socket_id);

    // core id (we add the socket_id to make it unique)
    std::ifstream core_id_file(topology / "core_id");
    if (!core_id_file.is_open()) {
      return cpu_topology{};
    }

    std::string core_id;
    core_id_file >> core_id;
    core_ids.insert(socket_id + "_" + core_id);
  }

  cpu_topology topo;
  topo.n_sockets          = package_ids.size();
  topo.procs_per_socket   = core_ids.size() / topo.n_sockets;
  topo.threads_per_socket = n_threads / topo.n_sockets;
  return topo;
}

static cpu_topology topology = init_cpu_topology();

std::optional<std::string> read_cpu_model_lscpu() {
  FILE* f = popen("lscpu --parse=MODELNAME 2>/dev/null", "r");
  if (!f) {
    return std::nullopt;
  }

  // We don't expect cpu model names to be longer that 1024 characters
  char buffer[1024];
  buffer[0] = '#';

  // lscpu puts some comments on the first lines of output
  while (buffer[0] == '#') {
    if (!std::fgets(buffer, 1024, f)) {
      return std::nullopt;
    }
  }

  pclose(f);

  std::string model_name(buffer);
  if (model_name.back() == '\n') {
    model_name.pop_back();
  }
  return model_name;
}

// Extract a value from /proc/cpuinfo
std::optional<std::string> get_cpu_info_str(const char* key) {
  std::ifstream cpu_info("/proc/cpuinfo");

  if (!cpu_info.is_open()) {
    return std::nullopt;
  }

  std::string value;
  std::string name;

  while (std::getline(cpu_info, name, ':')) {
    if (name.find(key) == 0) {
      if (std::isspace(cpu_info.peek())) {
        cpu_info.get();
      }
      std::getline(cpu_info, value);
      return value;
    }
    std::getline(cpu_info, value);
  }

  return std::nullopt;
}

// Extract a value from /etc/os-release
std::optional<std::string> get_os_release_str(const char* key) {
  std::ifstream os_release_file("/etc/os-release");

  if (!os_release_file.is_open()) {
    return std::nullopt;
  }

  std::string value;
  std::string name;

  while (std::getline(os_release_file, name, '=')) {
    std::getline(os_release_file, value);
    if (name == key) {
      if (value.front() == '"' && value.back() == '"') {
        return value.substr(1, value.size() - 2);
      }
      return value;
    }
  }

  return std::nullopt;
}

size_t get_physical_socket_count() { return topology.n_sockets; }

size_t get_core_count_per_socket() { return topology.procs_per_socket; }

size_t get_thread_count_per_socket() { return topology.threads_per_socket; }

std::string get_cpu_model_name() {
  // NOTE: /proc/cpuinfo on arm does not provide the CPU model name, lscpu on
  // the other hand seems to be reliable. If it fails we fall back to reading
  // from /proc/cpuinfo
  std::optional<std::string> cpu_model = read_cpu_model_lscpu();
  if (cpu_model.has_value()) {
    return cpu_model.value();
  } else {
    return get_cpu_info_str("model name").value_or("ERROR");
  }
}

std::string get_sys_name() {
  return get_os_release_str("PRETTY_NAME").value_or("ERROR");
}

std::string get_sys_type() {
  return get_proc_sys_value("kernel/ostype").value_or("ERROR");
}

std::string get_kernel_version() {
  return get_proc_sys_value("kernel/osrelease").value_or("ERROR");
}

}  // namespace cexa::experimental
// NOLINTEND(misc-definitions-in-headers)

#endif
