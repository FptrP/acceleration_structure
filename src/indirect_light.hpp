#ifndef INDIRECT_LIGHT_HPP_INCLUDED
#define INDIRECT_LIGHT_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene_renderer.hpp"
#include "imgui_pass.hpp"

struct IndirectLight {
  IndirectLight(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);
  void run(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, rendergraph::ImageResourceId diffuse_light, const DrawTAAParams &params);

  rendergraph::ImageResourceId get() const { return indirect_light_raw; }
private:
  rendergraph::ImageResourceId indirect_light_raw;
  gpu::ComputePipeline trace_software_pipeline;
  
  gpu::BufferPtr random_samples;
};

#endif