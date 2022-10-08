#ifndef GTAO_HPP_INCLUDED
#define GTAO_HPP_INCLUDED

#include "scene/camera.hpp"
#include "rendergraph/rendergraph.hpp"
#include "scene_renderer.hpp"

#define GTAO_TRACE_SAMPLES 0

rendergraph::ImageResourceId create_gtao_texture(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);

struct GTAOParams {
  glm::mat4 normal_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

struct GTAORTParams {
  glm::mat4 camera_to_world;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

struct GTAOReprojection {
  glm::mat4 camera_to_prev_frame;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

struct GTAO {
  GTAO(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height, bool use_ray_query, bool half_res = true, int pattern_n = 2);

  void add_main_pass(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal,
    rendergraph::ImageResourceId material,
    rendergraph::ImageResourceId preintegrated_pdf);
  
  void add_main_rt_pass(
    rendergraph::RenderGraph &graph,
    const GTAORTParams &params,
    VkAccelerationStructureKHR tlas,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal);
  
  void add_main_pass_graphics(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal);
  
  void add_filter_pass(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth);

  void add_reprojection_pass(
    rendergraph::RenderGraph &graph,
    const GTAOReprojection &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId prev_depth);

  void add_accumulate_pass(
    rendergraph::RenderGraph &graph,
    const DrawTAAParams &params,
    const Gbuffer &gbuffer);

  void deinterleave_depth(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth);
  void add_main_pass_deinterleaved(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId normal);

  void draw_ui();

  void remap(rendergraph::RenderGraph &graph) {
    graph.remap(accumulated_history, accumulated_ao);
  }

  rendergraph::ImageResourceId raw; //output of main pass
  rendergraph::ImageResourceId filtered; //output of filter pass
  rendergraph::ImageResourceId prev_frame; //previous frame
  rendergraph::ImageResourceId output; //final
  rendergraph::ImageResourceId accumulated_ao;
  rendergraph::ImageResourceId accumulated_history;
  rendergraph::ImageResourceId deinterleaved_depth;

private:

  gpu::GraphicsPipeline main_pipeline_gfx;
  gpu::ComputePipeline main_pipeline;
  gpu::GraphicsPipeline rt_main_pipeline;
  gpu::ComputePipeline filter_pipeline;
  gpu::ComputePipeline reproject_pipeline;
  gpu::ComputePipeline accumulate_pipeline;

  gpu::ComputePipeline deinterleave_pipeline;
  gpu::ComputePipeline main_deinterleaved_pipeline;
  
  int deinterleave_n = 2;
  uint32_t depth_lod = 0;

  gpu::BufferPtr random_vectors;
  bool mis_gtao = true;
  bool two_directions = false;
  bool only_reflections = false;
  bool clear_history = false;
  float weight_ratio = 1.0;

  uint32_t frame_count = 0;

  VkSampler sampler;
};

#endif