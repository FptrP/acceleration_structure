#ifndef CONTACT_SHADOWS_HPP_INCLUDED
#define CONTACT_SHADOWS_HPP_INCLUDED

#include "gpu/gpu.hpp"
#include "rendergraph/rendergraph.hpp"
#include "scene_renderer.hpp"

struct ContactShadows {

  void init(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h);
  
  void run(rendergraph::RenderGraph &graph, const DrawTAAParams &params, LightsManager &lights, rendergraph::ImageResourceId depth, VkAccelerationStructureKHR acc_struct = nullptr, bool depth_as = false);


  rendergraph::ImageResourceId get_output() const {
    return contact_shadows_raw;
  }

private:
  rendergraph::ImageResourceId contact_shadows_raw;

  gpu::ComputePipeline shadows_software;
  gpu::ComputePipeline shadows_hardware;
  gpu::ComputePipeline shadows_hardware_depth;

  void run_software(rendergraph::RenderGraph &graph, const DrawTAAParams &params, LightsManager &lights, rendergraph::ImageResourceId depth);
  void run_hardware(rendergraph::RenderGraph &graph, const DrawTAAParams &params, LightsManager &lights, VkAccelerationStructureKHR acc_struct, rendergraph::ImageResourceId depth, bool depth_as);
};

#endif