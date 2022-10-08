#ifndef TRACE_SAMPLES_HPP_INCLUDED
#define TRACE_SAMPLES_HPP_INCLUDED

#include <memory>
#include "rendergraph/rendergraph.hpp"
#include "util_passes.hpp"

struct SamplesMarker {

  static void init(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h) {
    instance.reset(new SamplesMarker{});
    gpu::ImageInfo info {VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
    auto usage = VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    instance->handle = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);

  }
  
  static void clear(rendergraph::RenderGraph &graph) {
    VkClearColorValue val {};
    val.uint32[0] = val.uint32[1] = val.uint32[2] = val.uint32[3] = 0;

    clear_color(graph, get_image(), val);
  }
  
  static rendergraph::ImageResourceId get_image() {
    return instance->handle;
  }

private:
  rendergraph::ImageResourceId handle;

  static std::unique_ptr<SamplesMarker> instance;
};



#endif