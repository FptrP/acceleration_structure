#include "gtao.hpp"
#include <cstdlib>
#include <random>
#include <cstring>
#include <iostream>

#include "imgui_pass.hpp"
#include "trace_samples.hpp"

rendergraph::ImageResourceId create_gtao_texture(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo info {VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  return graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
}

static gpu::BufferPtr create_random_vectors(uint32_t vectors_count);

GTAO::GTAO(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height, bool use_ray_query, bool half_res, int pattern_n)
  : deinterleave_n {pattern_n}
{
  if (half_res) {
    width /= 2;
    height /= 2;
    depth_lod = 1; 
  }

  gpu::ImageInfo info {VK_FORMAT_R16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo info_raw {VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  auto usage = VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;

  raw = graph.create_image(VK_IMAGE_TYPE_2D, info_raw, VK_IMAGE_TILING_OPTIMAL, usage|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  filtered = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  prev_frame = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  output = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);

  random_vectors = create_random_vectors(64);
  
  info.format = VK_FORMAT_R16G16_SFLOAT;
  accumulated_ao = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  accumulated_history = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);

  uint32_t pattern_step = 1u << (uint32_t)pattern_n;

  info.format = VK_FORMAT_R32_SFLOAT;
  info.width = width/pattern_step;
  info.height = height/pattern_step;
  info.array_layers = pattern_step * pattern_step;
  deinterleaved_depth = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);

  main_pipeline = gpu::create_compute_pipeline();
  main_pipeline.set_program("gtao_compute_main");

  if (use_ray_query) {
    rt_main_pipeline = gpu::create_graphics_pipeline();
    rt_main_pipeline.set_program("gtao_rt_main");
    rt_main_pipeline.set_registers({});
    rt_main_pipeline.set_vertex_input({});
    rt_main_pipeline.set_rendersubpass({false, {graph.get_descriptor(raw).format}});
  }
  
  filter_pipeline = gpu::create_compute_pipeline();
  filter_pipeline.set_program("gtao_filter");

  reproject_pipeline = gpu::create_compute_pipeline();
  reproject_pipeline.set_program("gtao_reproject");

  main_pipeline_gfx = gpu::create_graphics_pipeline();
  main_pipeline_gfx.set_program("gtao_main");
  main_pipeline_gfx.set_registers({});
  main_pipeline_gfx.set_vertex_input({});
  main_pipeline_gfx.set_rendersubpass({false, {graph.get_descriptor(raw).format}});

  accumulate_pipeline = gpu::create_compute_pipeline();
  accumulate_pipeline.set_program("gtao_accumulate");

  deinterleave_pipeline = gpu::create_compute_pipeline();
  deinterleave_pipeline.set_program("deinterleave_depth");

  main_deinterleaved_pipeline = gpu::create_compute_pipeline();
  main_deinterleaved_pipeline.set_program("main_deinterleaved");

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
}

void GTAO::add_main_pass(
  rendergraph::RenderGraph &graph,
  const GTAOParams &params,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId normal,
  rendergraph::ImageResourceId material,
  rendergraph::ImageResourceId preintegrated_pdf)
{

  struct PassData {
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId norm;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId pdf;
  };

  struct PushConsts {
    float base_angle;
    float weight_ratio;
    uint32_t use_mis;
    uint32_t two_directions;
    uint32_t reflections_only;
  };

  const float angle_offsets[] {60.f, 300.f, 180.f, 240.f, 120.f, 0.f, 300.f, 60.f, 180.f, 120.f, 240.f, 0.f};
  float base_angle = angle_offsets[frame_count % (sizeof(angle_offsets)/sizeof(float))]/360.f;
  base_angle += rand()/float(RAND_MAX) - 0.5;

  PushConsts push_consts {base_angle, weight_ratio, mis_gtao, two_directions? 255u : 0u, only_reflections? 255u : 0u};

  frame_count += 1;
  
  graph.add_task<PassData>("GTAO_main",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.norm = builder.sample_image(normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.pdf = builder.sample_image(preintegrated_pdf, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(raw, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAOParams>();
      *block.ptr = params;

      auto set = resources.allocate_set(main_pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::UBOBinding {1, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {2, resources.get_view(input.norm), sampler},
        gpu::TextureBinding {3, resources.get_view(input.material), sampler},
        gpu::TextureBinding {4, resources.get_view(input.pdf), sampler},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)}
      );

      const auto &extent = resources.get_image(input.out)->get_extent();

      cmd.bind_pipeline(main_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {block.offset});
      cmd.push_constants_compute(0, sizeof(push_consts), &push_consts);
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });

}

void GTAO::add_main_rt_pass(
    rendergraph::RenderGraph &graph,
    const GTAORTParams &params,
    VkAccelerationStructureKHR tlas,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal)
{
  struct PassData {
    rendergraph::ImageViewId rt;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId norm;
  };

  float base_angle = rand()/float(RAND_MAX) - 0.5;

  graph.add_task<PassData>("GTAO_rt_main",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.norm = builder.sample_image(normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.rt = builder.use_color_attachment(raw, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAORTParams>();
      *block.ptr = params;

      auto set = resources.allocate_set(rt_main_pipeline, 0);
    
      gpu::write_set(set,
        gpu::UBOBinding {0, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {1, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.norm), sampler},
        gpu::AccelerationStructBinding {3, tlas},
        gpu::UBOBinding {4, random_vectors}
      );

      const auto &extent = resources.get_image(input.rt)->get_extent();

      cmd.set_framebuffer(extent.width, extent.height, {resources.get_image_range(input.rt)});
      cmd.bind_pipeline(rt_main_pipeline);
      cmd.bind_viewport(0.f, 0.f, float(extent.width), float(extent.height), 0.f, 1.f);
      cmd.bind_scissors(0, 0, extent.width, extent.height);
      cmd.push_constants_graphics(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &base_angle);
      cmd.bind_descriptors_graphics(0, {set}, {block.offset, 0});
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    });
}

void GTAO::add_filter_pass(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth)
{

  struct PassData {
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId raw_gtao;
  };
  
  struct FilterData {
    float znear;
    float zfar;
  };

  FilterData filter_params {params.znear, params.zfar};

  graph.add_task<PassData>("GTAO_filter",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.raw_gtao = builder.sample_image(raw, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(filtered, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(filter_pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.raw_gtao), sampler},
        gpu::StorageTextureBinding {2, resources.get_view(input.out)}
      );

      const auto &extent = resources.get_image(input.out)->get_extent();

      cmd.bind_pipeline(filter_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {});
      cmd.push_constants_compute(0, sizeof(filter_params), &filter_params);
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });
}

void GTAO::add_reprojection_pass(
    rendergraph::RenderGraph &graph,
    const GTAOReprojection &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId prev_depth)
{
  struct PassData {
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId gtao;
    rendergraph::ImageViewId prev_gtao;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId prev_depth;
  };

  graph.add_task<PassData>("GTAO_reproject",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.prev_depth = builder.sample_image(prev_depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.gtao = builder.sample_image(filtered, VK_SHADER_STAGE_COMPUTE_BIT);
      input.prev_gtao = builder.sample_image(prev_frame, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(output, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAOReprojection>();
      *block.ptr = params;

      auto set = resources.allocate_set(reproject_pipeline, 0);
    
      gpu::write_set(set,
        gpu::UBOBinding {0, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {1, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.prev_depth), sampler},
        gpu::TextureBinding {3, resources.get_view(input.gtao), sampler},
        gpu::TextureBinding {4, resources.get_view(input.prev_gtao), sampler},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)}
      );

      const auto &extent = resources.get_image(input.out)->get_extent();

      cmd.bind_pipeline(reproject_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {block.offset});
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });
}

void GTAO::add_accumulate_pass(
    rendergraph::RenderGraph &graph,
    const DrawTAAParams &params,
    const Gbuffer &gbuffer)
{
  struct PassData {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId prev_depth;
    rendergraph::ImageViewId gtao;
    rendergraph::ImageViewId accumulated_ao;
    rendergraph::ImageViewId velocity;
    rendergraph::ImageViewId history;
  };

  struct AccumConstants {
    glm::mat4 inverse_camera;
    glm::mat4 prev_inverse_camera;
    glm::mat4 mvp;
    glm::vec4 fovy_aspect_znear_zfar;
  };

  struct PushConstants {
    uint32_t clear_history;
  };

  AccumConstants constants {glm::inverse(params.camera), glm::inverse(params.prev_camera), params.mvp, params.fovy_aspect_znear_zfar};
  PushConstants pc {clear_history? 1u : 0u};
  clear_history = false;

  graph.add_task<PassData>("GTAO_accumulate",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(gbuffer.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.prev_depth = builder.sample_image(gbuffer.prev_depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.gtao = builder.sample_image(filtered, VK_SHADER_STAGE_COMPUTE_BIT);
      input.accumulated_ao = builder.use_storage_image(accumulated_ao, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      input.velocity = builder.sample_image(gbuffer.downsampled_velocity_vectors, VK_SHADER_STAGE_COMPUTE_BIT);
      input.history = builder.sample_image(accumulated_history, VK_SHADER_STAGE_COMPUTE_BIT);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){

      auto set = resources.allocate_set(accumulate_pipeline, 0);
      auto blk = cmd.allocate_ubo<AccumConstants>();
      *blk.ptr = constants;

      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.prev_depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.gtao), sampler},
        gpu::StorageTextureBinding {3, resources.get_view(input.accumulated_ao)},
        gpu::TextureBinding {4, resources.get_view(input.velocity), sampler},
        gpu::TextureBinding {5, resources.get_view(input.history), sampler},
        gpu::UBOBinding {6, cmd.get_ubo_pool(), blk}
      );

      const auto &extent = resources.get_image(input.accumulated_ao)->get_extent();

      cmd.bind_pipeline(accumulate_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset});
      cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch((extent.width + 7)/8, (extent.height + 3)/4, 1);
    });
}

void GTAO::add_main_pass_graphics(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal)
{
  struct PushConstants {
    float angle_offset;
  };

  struct PassData {
    rendergraph::ImageViewId rt;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId norm;
  #if GTAO_TRACE_SAMPLES
    rendergraph::ImageViewId samples_map;
  #endif
  };

  const float angle_offsets[] {60.f, 300.f, 180.f, 240.f, 120.f, 0.f, 300.f, 60.f, 180.f, 120.f, 240.f, 0.f};
  float base_angle = angle_offsets[frame_count % (sizeof(angle_offsets)/sizeof(float))]/360.f;
  base_angle += rand()/float(RAND_MAX) - 0.5;

  PushConstants constants {base_angle};

  frame_count += 1;

  graph.add_task<PassData>("GTAO",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.norm = builder.sample_image(normal, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.rt = builder.use_color_attachment(raw, 0, 0);
    #if GTAO_TRACE_SAMPLES
      input.samples_map = builder.use_storage_image(SamplesMarker::get_image(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, 0);
    #endif
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAOParams>();
      *block.ptr = params;

      auto set = resources.allocate_set(main_pipeline_gfx, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::UBOBinding {1, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {2, resources.get_view(input.norm), sampler});
      
      #if GTAO_TRACE_SAMPLES
        gpu::write_set(set, 
          gpu::StorageTextureBinding {7, resources.get_view(input.samples_map)});
      #endif
      const auto &image_info = resources.get_image(input.rt)->get_extent();
      auto w = image_info.width;
      auto h = image_info.height;

      cmd.set_framebuffer(w, h, {resources.get_image_range(input.rt)});
      cmd.bind_pipeline(main_pipeline_gfx);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_descriptors_graphics(0, {set}, {block.offset});
      cmd.push_constants_graphics(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants), &constants);
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    });
}

static gpu::BufferPtr create_random_vectors(uint32_t vectors_count) {
  std::uniform_real_distribution<float> random_floats {0.0, 1.0};
  std::default_random_engine generator;
  std::vector<glm::vec4> random_vectors;
  random_vectors.reserve(vectors_count);

  while (random_vectors.size() < vectors_count) {
    float x = random_floats(generator) * 2.0 - 1.0;
    float y = random_floats(generator) * 2.0 - 1.0;
    float z = random_floats(generator);

    glm::vec4 dir {x, y, z, 0.f};
    float length = glm::length(dir);
    if (length <= 0.00001 || length > 1.f) {
      continue;
    }

    dir /= length;
    random_vectors.push_back(dir);
  }

  auto buffer_size = sizeof(glm::vec4) * random_vectors.size();
  
  
  auto vec_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  std::memcpy(vec_buffer->get_mapped_ptr(), random_vectors.data(), buffer_size);

  return vec_buffer;
}

void GTAO::deinterleave_depth(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth) {
  struct PassData {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId out;
  };

  graph.add_task<PassData>("GTAO_deinterleave",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_lod, 1, 0, 1);
      input.out = builder.use_storage_image_array(deinterleaved_depth, VK_SHADER_STAGE_COMPUTE_BIT);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){

      auto set = resources.allocate_set(deinterleave_pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::StorageTextureBinding {1, resources.get_view(input.out)}
      );

      const auto &extent = resources.get_image(input.out)->get_extent();

      cmd.bind_pipeline(deinterleave_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {});
      cmd.push_constants_compute(0, sizeof(deinterleave_n), &deinterleave_n);
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });
}

void GTAO::add_main_pass_deinterleaved(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId normal)
{
  struct PushConstants {
    int pattern_n;
    uint32_t layer;
    float angle_offset;
  };

  struct PassData {
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId norm;
  };

  const float angle_offsets[] {60.f, 300.f, 180.f, 240.f, 120.f, 0.f, 300.f, 60.f, 180.f, 120.f, 240.f, 0.f};
  float base_angle = angle_offsets[frame_count % (sizeof(angle_offsets)/sizeof(float))]/360.f;
  base_angle += rand()/float(RAND_MAX) - 0.5;

  frame_count += 1;

  graph.add_task<PassData>("GTAO_deinterleaved",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(deinterleaved_depth, VK_SHADER_STAGE_COMPUTE_BIT);
      input.norm = builder.sample_image(normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(raw, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAOParams>();
      *block.ptr = params;

      auto set = resources.allocate_set(main_deinterleaved_pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::UBOBinding {1, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {2, resources.get_view(input.norm), sampler},
        gpu::StorageTextureBinding {3, resources.get_view(input.out)}
      );
      const auto &info = resources.get_image(input.out);
      const auto &extent = info->get_extent();

      cmd.bind_pipeline(main_deinterleaved_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {block.offset});
      for (uint32_t i = 0; i < info->get_array_layers(); i++) {
        PushConstants pc {deinterleave_n, i, base_angle};
        cmd.push_constants_compute(0, sizeof(pc), &pc);
        cmd.dispatch(extent.width/8, extent.height/4, 1);
      }
    });
}

void GTAO::draw_ui() {
  ImGui::Begin("GTAO");
  ImGui::Checkbox("Enable MIS", &mis_gtao);
  ImGui::Checkbox("Use 2 directions", &two_directions);
  ImGui::Checkbox("Only reflections ao", &only_reflections);
  ImGui::SliderFloat("Weight ratio", &weight_ratio, 1.0, 5.0);
  clear_history = ImGui::Button("Clear history");
  ImGui::End();
}