#include "indirect_light.hpp"
#include <random>

const uint32_t NUM_RANDOM_VECTORS = 64;

IndirectLight::IndirectLight(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo image_info {VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width/2, height/2};
  indirect_light_raw = graph.create_image(VK_IMAGE_TYPE_2D, image_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  trace_software_pipeline = gpu::create_compute_pipeline("indirect_light_trace_software");
  random_samples = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(glm::vec4) * NUM_RANDOM_VECTORS, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  std::vector<glm::vec4> samples;
  samples.reserve(NUM_RANDOM_VECTORS);
  
  std::random_device rd {};
  std::mt19937 engine { rd() };
  std::uniform_real_distribution<float> gen {-1.f, 1.f}; 

  while (samples.size() < NUM_RANDOM_VECTORS) {
    glm::vec4 v {gen(engine), gen(engine), gen(engine), 0.f};
    if (v.z > 0.f && glm::length(glm::vec3(v)) <= 1.f) {
      samples.push_back(v);
    }
  }

  memcpy(random_samples->get_mapped_ptr(), samples.data(), sizeof(samples[0]) * samples.size());

}

void IndirectLight::run(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, rendergraph::ImageResourceId diffuse_light, const DrawTAAParams &params) {
  struct Input {
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId diffuse;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId out;
  };

  struct Constants {
    glm::mat4 normal_mat;
  
    float fovy;
    float aspect;
    float znear;
    float zfar;

    uint32_t random_seed;
    float rotation_offset;
  };

  uint32_t rand_seed = rand();
  float rand_angle = 2.f * glm::pi<float>() * float(rand())/RAND_MAX;

  Constants consts {glm::transpose(glm::inverse(params.camera)), params.fovy_aspect_znear_zfar.x, params.fovy_aspect_znear_zfar.y, params.fovy_aspect_znear_zfar.z, params.fovy_aspect_znear_zfar.w, rand_seed, rand_angle};

  graph.add_task<Input>("IndirectLightTraceSoftware", 
  [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
    input.albedo = builder.sample_image(gbuffer.albedo, VK_SHADER_STAGE_COMPUTE_BIT);
    input.diffuse = builder.sample_image(diffuse_light, VK_SHADER_STAGE_COMPUTE_BIT);
    input.depth = builder.sample_image(gbuffer.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1, 0, 1);
    input.normal = builder.sample_image(gbuffer.downsampled_normals, VK_SHADER_STAGE_COMPUTE_BIT);
    input.out = builder.use_storage_image(indirect_light_raw, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
  },
  [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
    auto set = resources.allocate_set(trace_software_pipeline, 0);
    auto blk = cmd.allocate_ubo<Constants>();
    *blk.ptr = consts;

    auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

    gpu::write_set(set, 
      gpu::TextureBinding {0, resources.get_view(input.albedo), sampler},
      gpu::TextureBinding {1, resources.get_view(input.diffuse), sampler},
      gpu::TextureBinding {2, resources.get_view(input.depth), sampler},
      gpu::TextureBinding {3, resources.get_view(input.normal), sampler},
      gpu::StorageTextureBinding {4, resources.get_view(input.out)},
      gpu::UBOBinding {5, cmd.get_ubo_pool(), blk},
      gpu::UBOBinding {6, random_samples});
      
    auto ext = resources.get_image(input.out)->get_extent();
    cmd.bind_pipeline(trace_software_pipeline);
    cmd.bind_descriptors_compute(0, {set}, {blk.offset, 0});
    //cmd.push_constants_compute(0, sizeof(push_consts), &push_consts);
    cmd.dispatch((ext.width + 7)/8, (ext.height + 3)/4, 1);
  });

}