#ifndef GPU_RESOURCE_INFO_HPP_INCLUDED
#define GPU_RESOURCE_INFO_HPP_INCLUDED

#include <algorithm>
#include <unordered_map>

#include "common.hpp"
#include "driver.hpp"

namespace gpu {
  struct ImageViewRange {
    VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D;
    VkImageAspectFlags aspect = 0;
    uint32_t base_mip = 0;
    uint32_t mips_count = 1;
    uint32_t base_layer = 0;
    uint32_t layers_count = 1;
    
    ImageViewRange() {}

    ImageViewRange(VkImageViewType type_, VkImageAspectFlags aspect_, uint32_t base_mip_, uint32_t mips_count_, uint32_t base_layer_, uint32_t layers_count_)
      : type {type_}, aspect {aspect_}, base_mip {base_mip_}, mips_count {mips_count_}, base_layer {base_layer_}, layers_count {layers_count_} {}
    
    ImageViewRange(VkImageViewType type_, uint32_t base_mip_, uint32_t mips_count_, uint32_t base_layer_, uint32_t layers_count_)
      : ImageViewRange(type_, 0, base_mip_, mips_count_,  base_layer_, layers_count_) {}


    bool operator==(const ImageViewRange &o) const {
      return (type == o.type)
        && (aspect == o.aspect)
        && (base_mip == o.base_mip)
        && (mips_count == o.mips_count)
        && (base_layer == o.base_layer)
        && (layers_count == o.layers_count); 
    }

    bool operator!=(const ImageViewRange &o) const {
      return (type != o.type)
        || (aspect != o.aspect)
        || (base_mip != o.base_mip)
        || (mips_count != o.mips_count)
        || (base_layer != o.base_layer)
        || (layers_count != o.layers_count); 
    }
  };

  inline ImageViewRange make_image_range2D(uint32_t base_mip, uint32_t mip_count, uint32_t base_layer = 0, uint32_t layer_count = 1) {
    return ImageViewRange {VK_IMAGE_VIEW_TYPE_2D, base_mip, mip_count, base_layer, layer_count};
  }
}

namespace std {
  template <>
  struct hash<gpu::ImageViewRange> {

    size_t operator()(const gpu::ImageViewRange &key) const {
      size_t h = 0;
      h ^= size_t(key.type);
      h ^= key.aspect << 1;
      h ^= key.base_layer << 2;
      h ^= key.layers_count << 3;
      h ^= key.base_mip << 4;
      h ^= key.mips_count << 5;
      return h;  
    }
  };
}

#endif