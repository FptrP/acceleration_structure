#ifndef SCENE_RENDERER_HPP_INCLUDED
#define SCENE_RENDERER_HPP_INCLUDED

#include "scene/scene.hpp"
#include "gpu/gpu.hpp"
#include "rendergraph/rendergraph.hpp"

#include <optional>
#include <memory>

struct Gbuffer {
  Gbuffer(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);

  rendergraph::ImageResourceId albedo;
  rendergraph::ImageResourceId normal;
  rendergraph::ImageResourceId downsampled_normals;
  rendergraph::ImageResourceId material;
  rendergraph::ImageResourceId depth;
  rendergraph::ImageResourceId prev_depth;
  rendergraph::ImageResourceId velocity_vectors;
  rendergraph::ImageResourceId downsampled_velocity_vectors;
  rendergraph::ImageResourceId triangle_id;

  uint32_t w, h;
};

struct DrawTAAParams {
  glm::mat4 mvp;
  glm::mat4 prev_mvp;
  glm::mat4 camera;
  glm::mat4 prev_camera;
  glm::vec4 jitter;
  glm::vec4 fovy_aspect_znear_zfar;
};

constexpr uint32_t MAX_DRAWCALLS = 2048u; 

struct SceneRenderer {
  SceneRenderer(scene::CompiledScene &s) : target {s} {}

  void init_pipeline(rendergraph::RenderGraph &graph, const Gbuffer &buffer);
  void update_scene();
  
  void draw_taa(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const DrawTAAParams &params);

  void rasterize_triange_id(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const DrawTAAParams &params);
  void reconstruct_gbuffer(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const DrawTAAParams &params);

  struct DrawCall {
    uint32_t transform;
    uint32_t primitive;
  };
  
  const std::vector<DrawCall> &get_drawcalls() const { return draw_calls; }
  
  rendergraph::BufferResourceId get_scene_transforms() const { return transform_buffer; }
  rendergraph::BufferResourceId get_drawcalls_buffer() const { return drawcall_buffer; }
  
  const scene::CompiledScene &get_target() const { return target; }

private:
  scene::CompiledScene &target;
  gpu::GraphicsPipeline opaque_taa_pipeline;
  gpu::GraphicsPipeline triangle_id_pipeline;
  gpu::ComputePipeline reconstruct_pipeline;

  gpu::ManagedDescriptorSet bindless_textures {}; 

  std::vector<std::pair<VkImageView, VkSampler>> scene_textures;
  std::vector<DrawCall> draw_calls;
  VkSampler sampler;
  VkSampler integer_sampler;

  rendergraph::BufferResourceId transform_buffer;
  rendergraph::BufferResourceId drawcall_buffer;
};


#endif