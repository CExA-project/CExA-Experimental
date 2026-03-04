#include "cexa_ArchInfo.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <ostream>

namespace cexa::experimental {

#if defined(UNIX) || defined(__unix__)

constexpr char CPUINFO_PATH[]    = "/proc/cpuinfo";
constexpr char OS_RELEASE_PATH[] = "/etc/os-release";
constexpr char SYS_PATH[]        = "/proc/sys/";

// Extract value from /proc/sys/ files
std::string get_proc_sys_value(const char* files) {
  std::ifstream proc_sys_file(SYS_PATH + std::string(files));
  std::string value(64, '\0');

  if (proc_sys_file.is_open()) {
    std::getline(proc_sys_file, value);
    if (proc_sys_file.fail()) {
      value[0] = '\0';
    }
  }
  return value;
}

// Extract a value from /proc/cpuinfo
size_t get_cpu_info_max_value(const char* key) {
  std::ifstream cpu_info(CPUINFO_PATH);
  std::string line;
  size_t max_value = 0;

  if (cpu_info.is_open()) {
    while (std::getline(cpu_info, line)) {
      if (line.find(key) != std::string::npos) {
        std::istringstream iss(line);
        size_t value = 0;
        // Remove key and extract int
        iss.ignore(256, ':');
        if (iss >> value) {
          max_value = max_value > value ? max_value : value;
        }
      }
    }
  }
  return max_value;
}

// Extract a value from /proc/cpuinfo
std::string get_cpu_info_str(const char* key) {
  std::ifstream cpu_info(CPUINFO_PATH);
  std::string value(128, '\0');
  std::string line;

  if (cpu_info.is_open()) {
    while (std::getline(cpu_info, line)) {
      if (line.find(key) != std::string::npos) {
        // Remove key and trim
        line.erase(0, line.find(':') + 1);
        line.erase(0, line.find_first_not_of(" \t"));
        return line;
      }
    }
  }
  return value;
}

// Extract a value from /etc/os-release
std::string get_os_release_str(const char* key) {
  std::ifstream os_release(OS_RELEASE_PATH);
  std::string value(128, '\0');
  std::string line;

  if (os_release.is_open()) {
    while (std::getline(os_release, line)) {
      if (line.find(key) != std::string::npos) {
        // Remove key and trim
        line.erase(0, line.find('=') + 1);
        line.erase(0, line.find_first_not_of("\""));
        line.erase(line.find_last_not_of("\"") + 1);
        return line;
      }
    }
  }
  return value;
}

size_t get_physical_socket_count() {
  return get_cpu_info_max_value("physical id") + 1;
}

size_t get_core_count_per_socket() {
  return get_cpu_info_max_value("cpu cores");
}

size_t get_thread_count_per_socket() {
  return get_cpu_info_max_value("processor") + 1;
}

std::string get_cpu_model_name() { return get_cpu_info_str("model name"); }

std::string get_sysname() { return get_os_release_str("PRETTY_NAME"); }

std::string get_sys_type() { return get_proc_sys_value("kernel/ostype"); }

std::string get_linux_kernel_version() {
  return get_proc_sys_value("kernel/osrelease");
}

#endif  // defined(UNIX) || defined(__unix__)

}  // namespace cexa::experimental

namespace Kokkos {

#if defined(UNIX) || defined(__unix__)

void print_cpu(std::ostream& ostream) {
  namespace CE = cexa::experimental;
  ostream << "CPU Model   : " << CE::get_cpu_model_name() << "\n"
          << "    Cores   : " << CE::get_core_count_per_socket() << "\n"
          << "    Threads : " << CE::get_thread_count_per_socket() << "\n"
          << "    Sockets : " << CE::get_physical_socket_count() << std::endl;
}

void print_os(std::ostream& ostream) {
  namespace CE = cexa::experimental;
  ostream << "OS Type   : " << CE::get_sys_type() << "\n"
          << "   Name   : " << CE::get_sysname() << "\n"
          << "   Kernel : " << CE::get_linux_kernel_version() << std::endl;
}

#elif defined(WIN32)

void print_cpu(std::ostream& ostream) {
  ostream << "CPU Model   : Unknown \n"
          << "    Cores   : Unknown \n"
          << "    Threads : Unknown \n"
          << "    Sockets : Unknown" << std::endl;
}

void print_os(std::ostream& ostream) {
  ostream << "OS Type   : Windows \n"
          << "   Name   : Unknown \n"
          << "   Kernel : Unknown" << std::endl;
}

#elif defined(__APPLE__)

void print_cpu(std::ostream& ostream) {
  ostream << "CPU Model   : Unknown \n"
          << "    Cores   : Unknown \n"
          << "    Threads : Unknown \n"
          << "    Sockets : Unknown" << std::endl;
}

void print_os(std::ostream& ostream) {
  ostream << "OS Type   : macOS \n"
          << "   Name   : Unknown \n"
          << "   Kernel : Unknown" << std::endl;
}

#endif

}  // namespace Kokkos
