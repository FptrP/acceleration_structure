#ifndef RTFX_HPP_INCLUDED
#define RTFX_HPP_INCLUDED

#include "depth_as.hpp"
#include "advanced_ssr.hpp"

struct RTReflections {
  RTReflections(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);
  
  void run(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const DepthAs &depth_as, const AdvancedSSRParams &params);

  rendergraph::ImageResourceId get_target() const {
    return result;
  }

private:
  gpu::ComputePipeline pipeline;
  rendergraph::ImageResourceId result;
  VkSampler sampler {nullptr}; 
};

#endif