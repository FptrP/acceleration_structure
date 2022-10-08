#ifndef RESOURCES_HPP_INCLUDED
#define RESOURCES_HPP_INCLUDED

#include "common.hpp"
#include "driver.hpp"
#include "resource_info.hpp"

#include <lib/vk_mem_alloc.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace gpu {

  enum class ImageCreateOptions {
    None,
    Cubemap,
    Array2D
  };

  struct ImageInfo {
    VkFormat format;
    VkImageAspectFlags aspect = 0;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;

    ImageInfo() {}
    ImageInfo(VkFormat fmt, VkImageAspectFlags aspect_flags, uint32_t w, uint32_t h)
      : format {fmt}, aspect {aspect_flags}, width {w}, height {h} {}
    
    ImageInfo(VkFormat fmt, VkImageAspectFlags aspect_flags, uint32_t w, uint32_t h, uint32_t d, uint32_t mips, uint32_t layers)
      : format {fmt}, aspect {aspect_flags}, width {w}, height {h}, depth {d}, mip_levels{mips}, array_layers{layers} {}


    const VkExtent3D extent3D() const { return {width, height, depth}; }
    const VkExtent2D extent2D() const { return {width, height}; }
  };

  struct AccelerationStructure {

  };
}

#endif