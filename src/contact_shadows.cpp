#include "contact_shadows.hpp"

#include <random>

void ContactShadows::init(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h) {
  gpu::ImageInfo info {VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
  contact_shadows_raw = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  shadows_software = gpu::create_compute_pipeline("contact_shadows_software");
  shadows_hardware = gpu::create_compute_pipeline("contact_shadows_hardware");
  shadows_hardware_depth = gpu::create_compute_pipeline("contact_shadows_hardware_depth");
}

struct ShadowConstants {
  glm::mat4 camera_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

void gen_sample_offsets(float *result, uint32_t num_samples) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
 
  std::normal_distribution<> d {0.f, 1.f};
 
  for (uint32_t i = 0; i < num_samples; i++) {
    result[i] = d(gen);
  }
}

struct RandomData {
  //uint32_t seed;
  float offsets[32];
};

void ContactShadows::run_software(rendergraph::RenderGraph &graph, const DrawTAAParams &params, LightsManager &lights, rendergraph::ImageResourceId depth) {
  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId out;
  };

  auto lights_buf = lights.get_buffer();

  ShadowConstants consts {
    params.camera,
    params.fovy_aspect_znear_zfar.x,
    params.fovy_aspect_znear_zfar.y,
    params.fovy_aspect_znear_zfar.z,
    params.fovy_aspect_znear_zfar.w
  };

  graph.add_task<Input>("ContactShadowsSoftware", 
  [&](Input &input, rendergraph::RenderGraphBuilder &builder){
    input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, builder.get_image_info(depth).mip_levels - 1, 0, 1);
    input.out = builder.use_storage_image(contact_shadows_raw, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    builder.use_uniform_buffer(lights_buf, VK_SHADER_STAGE_COMPUTE_BIT);
  },
  [=](Input &input, rendergraph::RenderResources &res, gpu::CmdContext &cmd) {
    auto ubo = cmd.allocate_ubo<ShadowConstants>();
    *ubo.ptr = consts;

    auto set = res.allocate_set(shadows_software, 0);
    gpu::write_set(set, 
      gpu::TextureBinding {0, res.get_view(input.depth), gpu::create_sampler(gpu::DEFAULT_SAMPLER)},
      gpu::StorageTextureBinding {1, res.get_view(input.out)},
      gpu::UBOBinding {2, cmd.get_ubo_pool(), ubo},
      gpu::UBOBinding {3, res.get_buffer(lights_buf)});

    auto extent = res.get_image(input.out)->get_extent();

    cmd.bind_pipeline(shadows_software);
    cmd.bind_descriptors_compute(0, {set}, {ubo.offset, 0});
    
    RandomData rd {};
      //rd.seed = rand();
    gen_sample_offsets(rd.offsets, 32);
    cmd.push_constants_compute(0, sizeof(rd), &rd);
    cmd.dispatch((extent.width + 7)/8, (extent.height + 3)/4, 1);
  });

}

void ContactShadows::run_hardware(rendergraph::RenderGraph &graph, const DrawTAAParams &params, LightsManager &lights, VkAccelerationStructureKHR acc_struct, rendergraph::ImageResourceId depth, bool depth_as) {
  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId out;
  };

  auto lights_buf = lights.get_buffer();

  ShadowConstants consts {
    params.camera,
    params.fovy_aspect_znear_zfar.x,
    params.fovy_aspect_znear_zfar.y,
    params.fovy_aspect_znear_zfar.z,
    params.fovy_aspect_znear_zfar.w
  };

  graph.add_task<Input>("ContactShadowsHardware", 
  [&](Input &input, rendergraph::RenderGraphBuilder &builder){
    input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1, 0, 1);
    input.out = builder.use_storage_image(contact_shadows_raw, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    builder.use_uniform_buffer(lights_buf, VK_SHADER_STAGE_COMPUTE_BIT);
  },
  [=](Input &input, rendergraph::RenderResources &res, gpu::CmdContext &cmd) {
    auto ubo = cmd.allocate_ubo<ShadowConstants>();
    *ubo.ptr = consts;
    
    auto &pipeline = depth_as? shadows_hardware_depth : shadows_hardware;

    auto set = res.allocate_set(pipeline, 0);
    gpu::write_set(set, 
      gpu::TextureBinding {0, res.get_view(input.depth), gpu::create_sampler(gpu::DEFAULT_SAMPLER)},
      gpu::StorageTextureBinding {1, res.get_view(input.out)},
      gpu::UBOBinding {2, cmd.get_ubo_pool(), ubo},
      gpu::AccelerationStructBinding {3, acc_struct},
      gpu::UBOBinding {4, res.get_buffer(lights_buf)});

    auto extent = res.get_image(input.out)->get_extent();

    cmd.bind_pipeline(pipeline);
    cmd.bind_descriptors_compute(0, {set}, {ubo.offset, 0});

    if (depth_as) {
      RandomData rd {};
      //rd.seed = rand();
      gen_sample_offsets(rd.offsets, 32);
      cmd.push_constants_compute(0, sizeof(rd), &rd);
    }

    cmd.dispatch((extent.width + 7)/8, (extent.height + 7)/8, 1);
  });
}

void ContactShadows::run(rendergraph::RenderGraph &graph, const DrawTAAParams &params, LightsManager &lights,  rendergraph::ImageResourceId depth, VkAccelerationStructureKHR acc_struct, bool depth_as) {
  if (acc_struct) {
    run_hardware(graph, params, lights, acc_struct, depth, depth_as);
  } else {
    run_software(graph, params, lights, depth);
  }
}