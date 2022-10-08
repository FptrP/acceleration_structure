#ifndef TAA_HPP_INCLUDED
#define TAA_HPP_INCLUDED

#include "scene/camera.hpp"
#include "rendergraph/rendergraph.hpp"
#include "scene_renderer.hpp"

struct TAA {
  TAA(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h);

  void run(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, rendergraph::ImageResourceId color, const DrawTAAParams &params);
  void remap_targets(rendergraph::RenderGraph &graph);

  rendergraph::ImageResourceId get_output() const { return target; }

private:
  rendergraph::ImageResourceId history;
  rendergraph::ImageResourceId target;
  gpu::ComputePipeline pipeline;
  VkSampler sampler;
};

#endif