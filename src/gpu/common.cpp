#include "common.hpp"

#include <sstream>

namespace gpu {
  void vk_check_error(VkResult result, const char *file, int line, const char *cmd) {
    if (result != VK_SUCCESS) {
      std::stringstream ss; 
      ss << "Error : " << file << ":" << line << " " << cmd << " : " << " VkResult == " << (int32_t)result;
      throw std::runtime_error {ss.str()}; 
    }
  }
}