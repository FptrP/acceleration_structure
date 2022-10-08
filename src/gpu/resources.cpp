#include "resources.hpp"
#include <stdexcept>

namespace gpu {
  static VkImageCreateFlags options_to_flags(ImageCreateOptions options) {
    switch (options) {
    case ImageCreateOptions::Array2D:
      return VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    case ImageCreateOptions::Cubemap:
      return VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    default:
      return 0;
    }
  }
}