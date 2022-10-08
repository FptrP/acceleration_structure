#ifndef DEFFERED_SHADING_HPP_INCLUDED
#define DEFFERED_SHADING_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene_renderer.hpp"
#include "imgui_pass.hpp"

struct DeferedShadingPass {

  DeferedShadingPass(rendergraph::RenderGraph &graph, SDL_Window *window);

  void update_params(const glm::mat4 &camera, const glm::mat4 &shadow, float fovy, float aspect, float znear, float zfar);
  
  void draw(rendergraph::RenderGraph &graph, 
    const Gbuffer &gbuffer,
    rendergraph::ImageResourceId shadow,
    rendergraph::ImageResourceId ssao,
    rendergraph::ImageResourceId brdf_tex,
    rendergraph::ImageResourceId reflections,
    rendergraph::ImageResourceId out_image);
  
  void draw_ui();

private:

  gpu::GraphicsPipeline pipeline;
  VkSampler sampler;
  rendergraph::BufferResourceId ubo_consts;

  glm::vec2 min_max_roughness {0.f, 1.f};
  bool only_ao = false;
};


#endif