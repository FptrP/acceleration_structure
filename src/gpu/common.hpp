#ifndef VKERROR_HPP_INCLUDED
#define VKERROR_HPP_INCLUDED

#include <cstdint>
#include <functional>

#include "lib/volk.h"

#define VKCHECK(expr) gpu::vk_check_error((expr), __FILE__, __LINE__, #expr) 

namespace gpu {
  void vk_check_error(VkResult result, const char *file, int line, const char *cmd);

  template <typename T>
  inline void hash_combine(std::size_t &s, const T &v) {
    std::hash<T> h;
    s ^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2); 
  }
}

#endif