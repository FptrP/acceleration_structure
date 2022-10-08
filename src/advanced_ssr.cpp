#include "advanced_ssr.hpp"
#include "imgui_pass.hpp"

#include <cstring>

#define HALTON_SEQ_SIZE 128

static float halton_elem(uint32_t index, uint32_t base) {
  float f = 1;
  float r = 0;
  uint32_t current = index;
  do
  {
    f = f / base;
    r = r + f * (current % base);
    current = glm::floor(current / float(base));
  } while (current > 0);
  
  return r;
}

std::vector<glm::vec4> halton23_seq(uint32_t count) {
  std::vector<glm::vec4> seq {};
  seq.reserve(count);

  for (uint32_t iter = 0; iter < count; iter++) {
    glm::vec4 elem {0, 0, 0, 0};
    elem.x = halton_elem(iter + 1, 2);
    elem.y = halton_elem(iter + 1, 3);
    seq.push_back(elem);
  }

  return seq;
}

AdvancedSSR::AdvancedSSR(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h) {
  trace_pass = gpu::create_compute_pipeline();
  trace_pass.set_program("sssr_trace");
  
  filter_pass = gpu::create_compute_pipeline();
  filter_pass.set_program("sssr_filter");

  blur_pass = gpu::create_compute_pipeline();
  blur_pass.set_program("sssr_blur");

  classification_pass = gpu::create_compute_pipeline();
  classification_pass.set_program("sssr_classification");

  trace_indirect_pass = gpu::create_compute_pipeline();
  trace_indirect_pass.set_program("sssr_trace_indirect");

  tile_regression = gpu::create_compute_pipeline("tile_regression");
  preintegrate_pass = gpu::create_compute_pipeline("pdf_preintegrate");
  preintegrate_brdf_pass = gpu::create_compute_pipeline("brdf_preintegrate");

  auto halton_samples = halton23_seq(HALTON_SEQ_SIZE);
  const uint64_t bytes = sizeof(halton_samples[0]) * HALTON_SEQ_SIZE;

  halton_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  std::memcpy(halton_buffer->get_mapped_ptr(), halton_samples.data(), bytes);
  
  gpu::ImageInfo rays_info {VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, w/2, h/2};
  rays = graph.create_image(VK_IMAGE_TYPE_2D, rays_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  rays_info.format = VK_FORMAT_R16_SFLOAT;
  rays_occlusion = graph.create_image(VK_IMAGE_TYPE_2D, rays_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  gpu::ImageInfo reflections_info {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, w/2, h/2};
  reflections = graph.create_image(VK_IMAGE_TYPE_2D, reflections_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  gpu::ImageInfo blurred_info {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, w/2, h/2};
  blurred_reflection = graph.create_image(VK_IMAGE_TYPE_2D, blurred_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);
  blurred_reflection_history = graph.create_image(VK_IMAGE_TYPE_2D, blurred_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);
  
  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  const auto indirect_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  reflective_indirect = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(VkDispatchIndirectCommand), indirect_usage);
  glossy_indirect = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(VkDispatchIndirectCommand), indirect_usage);
  
  const uint64_t tile_bytes = sizeof(int) * (w * h/64);
  reflective_tiles = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, tile_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  glossy_tiles = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, tile_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  gpu::ImageInfo tile_planes_info {VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, w/16, h/16};
  tile_planes = graph.create_image(VK_IMAGE_TYPE_2D, tile_planes_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  gpu::ImageInfo pdf_info {VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1024, 1024};
  preintegrated_pdf = graph.create_image(VK_IMAGE_TYPE_2D, pdf_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  gpu::ImageInfo brdf_info {VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1024, 1024};
  preintegrated_brdf = graph.create_image(VK_IMAGE_TYPE_2D, brdf_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);
}

void AdvancedSSR::preintegrate_pdf(rendergraph::RenderGraph &graph) {
  struct Input {
    rendergraph::ImageViewId out_pdf;
  };

  graph.add_task<Input>("SSR_preintegrate",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder){
      input.out_pdf = builder.use_storage_image(preintegrated_pdf, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      auto set = resources.allocate_set(preintegrate_pass, 0);
      gpu::write_set(set, 
        gpu::StorageTextureBinding {0, resources.get_view(input.out_pdf)});

      auto extent = resources.get_image(input.out_pdf)->get_extent();
      cmd.bind_pipeline(preintegrate_pass);
      cmd.bind_descriptors_compute(0, {set});
      cmd.dispatch((extent.width + 7)/8, (extent.height + 3)/4, 1);
    });
}

void AdvancedSSR::preintegrate_brdf(rendergraph::RenderGraph &graph) {
  struct Input {
    rendergraph::ImageViewId out_brdf;
  };

  graph.add_task<Input>("BRDF_preintegrate",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder){
      input.out_brdf = builder.use_storage_image(preintegrated_brdf, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      auto set = resources.allocate_set(preintegrate_brdf_pass, 0);
      gpu::write_set(set, 
        gpu::UBOBinding {0, halton_buffer},
        gpu::StorageTextureBinding {1, resources.get_view(input.out_brdf)});

      auto extent = resources.get_image(input.out_brdf)->get_extent();
      cmd.bind_pipeline(preintegrate_brdf_pass);
      cmd.bind_descriptors_compute(0, {set}, {0});
      cmd.dispatch((extent.width + 7)/8, (extent.height + 3)/4, 1);
    });
}

struct TraceParams {
  glm::mat4 normal_mat;
  uint32_t frame_random;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

void AdvancedSSR::run_trace_pass(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const Gbuffer &gbuff,
  rendergraph::ImageResourceId ssr_occlusion)
{
  TraceParams config {
    params.normal_mat,
    counter,
    params.fovy,
    params.aspect,
    params.znear,
    params.zfar
  };

  struct PushConstants {
    float max_roughness;  
  };

  PushConstants push_consts {settings.max_rougness};

  if (settings.update_random) {
    counter++;
    counter = counter % settings.max_accumulated_rays;
  }

  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId occlusion;
    rendergraph::ImageViewId preintegrated_pdf;
  };
  
  auto mips_count = graph.get_descriptor(gbuff.depth).mip_levels;

  graph.add_task<Input>("SSSR_trace",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.depth = builder.sample_image(gbuff.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, mips_count - 1, 0, 1);
      input.normal = builder.sample_image(gbuff.downsampled_normals, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(gbuff.material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(rays, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      input.occlusion = builder.use_storage_image(ssr_occlusion, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      input.preintegrated_pdf = builder.sample_image(preintegrated_pdf, VK_SHADER_STAGE_COMPUTE_BIT);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(trace_pass, 0);
      auto blk = cmd.allocate_ubo<TraceParams>();
      *blk.ptr = config;

      gpu::write_set(set, 
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.material), sampler},
        gpu::UBOBinding {3, cmd.get_ubo_pool(), blk},
        gpu::UBOBinding {4, halton_buffer},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)},
        gpu::StorageTextureBinding {6, resources.get_view(input.occlusion)},
        gpu::TextureBinding {7, resources.get_view(input.preintegrated_pdf), sampler});
      
      auto ext = resources.get_image(input.out)->get_extent();
      cmd.bind_pipeline(trace_pass);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset, 0});
      cmd.push_constants_compute(0, sizeof(push_consts), &push_consts);
      cmd.dispatch((ext.width + 7)/8, (ext.height + 7)/8, 1);
    });
}

void AdvancedSSR::run_trace_indirect_pass(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const Gbuffer &gbuff)
{
  TraceParams config {
    params.normal_mat,
    counter,
    params.fovy,
    params.aspect,
    params.znear,
    params.zfar
  };

  struct PushConstants {
    uint32_t reflection_type;
    float max_roughness;  
  };

  PushConstants push_consts {0, settings.max_rougness};

  if (settings.update_random) {
    counter++;
    counter = counter % settings.max_accumulated_rays;
  }

  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId out;
  };
  
  auto mips_count = graph.get_descriptor(gbuff.depth).mip_levels;

  graph.add_task<Input>("SSSR_trace",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.depth = builder.sample_image(gbuff.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, mips_count - 1, 0, 1);
      input.normal = builder.sample_image(gbuff.downsampled_normals, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(gbuff.material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(rays, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      
      builder.use_indirect_buffer(reflective_indirect);
      builder.use_indirect_buffer(glossy_indirect);
      builder.use_storage_buffer(reflective_tiles, VK_SHADER_STAGE_COMPUTE_BIT);
      builder.use_storage_buffer(glossy_tiles, VK_SHADER_STAGE_COMPUTE_BIT);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set_mirror = resources.allocate_set(trace_indirect_pass, 0);
      auto set_glossy = resources.allocate_set(trace_indirect_pass, 0);

      auto blk = cmd.allocate_ubo<TraceParams>();
      *blk.ptr = config;

      gpu::write_set(set_mirror, 
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.material), sampler},
        gpu::UBOBinding {3, cmd.get_ubo_pool(), blk},
        gpu::UBOBinding {4, halton_buffer},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)},
        gpu::SSBOBinding {6, resources.get_buffer(reflective_tiles)});
      
      gpu::write_set(set_glossy, 
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.material), sampler},
        gpu::UBOBinding {3, cmd.get_ubo_pool(), blk},
        gpu::UBOBinding {4, halton_buffer},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)},
        gpu::SSBOBinding {6, resources.get_buffer(glossy_tiles)});
      
      auto pc = push_consts;

      cmd.bind_pipeline(trace_indirect_pass);
      
      pc.reflection_type = 0; //mirror reflections
      cmd.bind_descriptors_compute(0, {set_mirror}, {blk.offset, 0});
      cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch_indirect(resources.get_buffer(reflective_indirect)->api_buffer());
      
      pc.reflection_type = 1; //glossy reflections
      cmd.bind_descriptors_compute(0, {set_glossy}, {blk.offset, 0});
      cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch_indirect(resources.get_buffer(glossy_indirect)->api_buffer());
    });
}

#define NORMALIZE_REFLECTIONS 1
#define ACCUMULATE_REFLECTIONS 2
#define BILATERAL_FILTER 4

void AdvancedSSR::run_filter_pass(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const Gbuffer &gbuff)
{
  TraceParams config {
    params.normal_mat,
    counter,
    params.fovy,
    params.aspect,
    params.znear,
    params.zfar
  };

  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId rays;
    rendergraph::ImageViewId reflection; 
  };

  struct PushConstants {
    uint32_t render_flags;
  };

  PushConstants pc {0u};
  pc.render_flags |= settings.normalize_reflections? NORMALIZE_REFLECTIONS : 0;
  pc.render_flags |= settings.accumulate_reflections? ACCUMULATE_REFLECTIONS : 0;
  pc.render_flags |= settings.bilateral_filter? BILATERAL_FILTER : 0;

  graph.add_task<Input>("SSSR_filter",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.depth = builder.sample_image(gbuff.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 10, 0, 1);
      input.normal = builder.sample_image(gbuff.normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.albedo = builder.sample_image(gbuff.albedo, VK_SHADER_STAGE_COMPUTE_BIT);
      input.rays = builder.sample_image(rays, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(gbuff.material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.reflection = builder.use_storage_image(reflections, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(filter_pass, 0);
      auto blk = cmd.allocate_ubo<TraceParams>();
      *blk.ptr = config;

      gpu::write_set(set, 
        gpu::TextureBinding {0, resources.get_view(input.rays), sampler},
        gpu::TextureBinding {1, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.albedo), sampler},
        gpu::TextureBinding {3, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {4, resources.get_view(input.material), sampler},
        gpu::StorageTextureBinding {5, resources.get_view(input.reflection)},
        gpu::UBOBinding {6, cmd.get_ubo_pool(), blk});
      
      auto ext = resources.get_image(input.reflection)->get_extent();
      cmd.bind_pipeline(filter_pass);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset});
      cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch((ext.width + 7)/8, (ext.height + 7)/8, 1);
    });
}

void AdvancedSSR::run_blur_pass(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const DrawTAAParams &taa_params,
  const Gbuffer &gbuff)
{
  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId reflections;
    rendergraph::ImageViewId history;
    rendergraph::ImageViewId velocity;
    rendergraph::ImageViewId history_depth;
    rendergraph::ImageViewId result;
  };
  
  struct PushConstants {
    float max_roughness;
    uint32_t accumulate;
    uint32_t disable_blur;
  };

  struct Params {
    glm::mat4 inverse_camera;
    glm::mat4 prev_inverse_camera;
    glm::vec4 fovy_aspect_znear_zfar;
  };

  PushConstants pc {settings.max_rougness, settings.accumulate_reflections, !settings.use_blur}; 
  Params buf {glm::inverse(taa_params.camera), glm::inverse(taa_params.prev_camera), taa_params.fovy_aspect_znear_zfar};

  graph.add_task<Input>("SSSR_blur",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(gbuff.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 10, 0, 1);
      input.normal = builder.sample_image(gbuff.normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.reflections = builder.sample_image(reflections, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(gbuff.material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.history = builder.sample_image(blurred_reflection_history, VK_SHADER_STAGE_COMPUTE_BIT);
      input.velocity = builder.sample_image(gbuff.downsampled_velocity_vectors, VK_SHADER_STAGE_COMPUTE_BIT);
      input.history_depth = builder.sample_image(gbuff.prev_depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 10, 0, 1);
      input.result = builder.use_storage_image(blurred_reflection, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      auto blk = cmd.allocate_ubo<Params>();
      *blk.ptr = buf;
      
      auto set = resources.allocate_set(blur_pass, 0);

      gpu::write_set(set, 
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.reflections), sampler},
        gpu::TextureBinding {3, resources.get_view(input.material), sampler},
        gpu::TextureBinding {4, resources.get_view(input.history), sampler},
        gpu::TextureBinding {5, resources.get_view(input.velocity), sampler},
        gpu::TextureBinding {6, resources.get_view(input.history_depth), sampler},
        gpu::StorageTextureBinding {7, resources.get_view(input.result)},
        gpu::UBOBinding {8, cmd.get_ubo_pool(), blk}
      );
      
      auto ext = resources.get_image(input.result)->get_extent();
      cmd.bind_pipeline(blur_pass);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset});
      cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch((ext.width + 7)/8, (ext.height + 7)/8, 1);
    });
}

void AdvancedSSR::clear_indirect_params(rendergraph::RenderGraph &graph) {
  struct Input {};
  graph.add_task<Input>("SSSR_Clear", 
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      builder.transfer_write(reflective_indirect);
      builder.transfer_write(glossy_indirect);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      VkDispatchIndirectCommand initial_dispath {0, 1, 1};
      cmd.update_buffer(resources.get_buffer(reflective_indirect)->api_buffer(), 0, initial_dispath);
      cmd.update_buffer(resources.get_buffer(glossy_indirect)->api_buffer(), 0, initial_dispath);
    });
}

void AdvancedSSR::run_classification_pass(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const Gbuffer &gbuff)
{
  struct Input {
    rendergraph::ImageViewId material_tex;
    //rendergraph::BufferResourceId
  };

  struct PushConstants {
    int width, height;
    float g_max_roughness;
    float g_glossy_value;
  };

  auto extent = graph.get_descriptor(rays).extent2D();
  PushConstants pc {int(extent.width), int(extent.height), settings.max_rougness, settings.glossy_roughness_value};

  graph.add_task<Input>("SSSR_Classification", 
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.material_tex = builder.sample_image(gbuff.material, VK_SHADER_STAGE_COMPUTE_BIT);
      builder.use_storage_buffer(reflective_indirect, VK_SHADER_STAGE_COMPUTE_BIT, false);
      builder.use_storage_buffer(glossy_indirect, VK_SHADER_STAGE_COMPUTE_BIT, false);
      builder.use_storage_buffer(reflective_tiles, VK_SHADER_STAGE_COMPUTE_BIT, false);
      builder.use_storage_buffer(glossy_tiles, VK_SHADER_STAGE_COMPUTE_BIT, false);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      auto set = resources.allocate_set(classification_pass, 0);
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.material_tex), sampler},
        gpu::SSBOBinding {1, resources.get_buffer(reflective_tiles)},
        gpu::SSBOBinding {2, resources.get_buffer(glossy_tiles)},
        gpu::SSBOBinding {3, resources.get_buffer(reflective_indirect)},
        gpu::SSBOBinding {4, resources.get_buffer(glossy_indirect)});

      cmd.bind_pipeline(classification_pass);
      cmd.bind_descriptors_compute(0, {set});
      cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch((extent.width + 7)/8, (extent.height + 7)/8, 1);
    });
}

void AdvancedSSR::run_tile_regression_pass(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const Gbuffer &gbuff)
{
  struct Input {
    rendergraph::ImageViewId depth_tex;
    rendergraph::ImageViewId planes_tex;
    //rendergraph::BufferResourceId
  };

  struct PushConstants {
    glm::mat4 camera_to_world;
    float fovy;
    float aspect;
    float znear;
    float zfar;
  };

  auto extent = graph.get_descriptor(gbuff.depth).extent2D();
  extent.width /= 2;
  extent.height /= 2;
  PushConstants pc {glm::transpose(params.normal_mat), params.fovy, params.aspect, params.znear, params.zfar};

  graph.add_task<Input>("SSSR_Tile_Regression", 
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.depth_tex = builder.sample_image(gbuff.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1, 0, 1);
      input.planes_tex = builder.use_storage_image(tile_planes, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      auto set = resources.allocate_set(tile_regression, 0);
      
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth_tex), sampler},
        gpu::StorageTextureBinding {1, resources.get_view(input.planes_tex)});

      cmd.bind_pipeline(tile_regression);
      cmd.bind_descriptors_compute(0, {set});
      cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch((extent.width + 7)/8, (extent.height + 7)/8, 1);
    });
}

void AdvancedSSR::run(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const DrawTAAParams &taa_params,
  const Gbuffer &gbuff,
  rendergraph::ImageResourceId ssr_occlusion)
{
  //clear_indirect_params(graph);
  //run_classification_pass(graph, params, gbuff);
  //run_tile_regression_pass(graph, params, gbuff);
  //run_trace_indirect_pass(graph, params, gbuff);
  run_trace_pass(graph, params, gbuff, ssr_occlusion);
  run_filter_pass(graph, params, gbuff);
  run_blur_pass(graph, params, taa_params, gbuff);
}

void AdvancedSSR::render_ui() {
  ImGui::Begin("SSSR");
  ImGui::SliderFloat("Max Roughness", &settings.max_rougness, 0.f, 1.f);
  ImGui::SliderFloat("Min glossy roughness", &settings.glossy_roughness_value, 0.f, 1.f);
  ImGui::SliderInt("Temporal rays", &settings.max_accumulated_rays, 1, 128);
  ImGui::Checkbox("Enable normalization", &settings.normalize_reflections);
  ImGui::Checkbox("Enable accumulation", &settings.accumulate_reflections);
  ImGui::Checkbox("Enable random rays", &settings.update_random);
  ImGui::Checkbox("Enable blur", &settings.use_blur);
  ImGui::Checkbox("Enable bilateral filter", &settings.bilateral_filter);
  ImGui::End();
}