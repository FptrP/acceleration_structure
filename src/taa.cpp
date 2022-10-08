#include "taa.hpp"

TAA::TAA(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h) {
  pipeline = gpu::create_compute_pipeline("taa_resolve");

  gpu::ImageInfo info {VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
  auto usage = VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  history = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  target = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
}

void TAA::run(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, rendergraph::ImageResourceId color, const DrawTAAParams &params) {
  struct PassData {
    rendergraph::ImageViewId history_color;
    rendergraph::ImageViewId history_depth;
    rendergraph::ImageViewId current_depth;
    rendergraph::ImageViewId velocity;
    rendergraph::ImageViewId color;
    rendergraph::ImageViewId out;
  };

  struct TAAParams {
    glm::mat4 inverse_camera;
    glm::mat4 prev_inverse_camera;
    glm::vec4 fovy_aspect_znear_zfar;
  };

  TAAParams consts {glm::inverse(params.camera), glm::inverse(params.prev_camera), params.fovy_aspect_znear_zfar};

  graph.add_task<PassData>("TAA",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.history_color = builder.sample_image(history, VK_SHADER_STAGE_COMPUTE_BIT);
      input.history_depth = builder.sample_image(gbuffer.prev_depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
      input.current_depth = builder.sample_image(gbuffer.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
      input.velocity = builder.sample_image(gbuffer.velocity_vectors, VK_SHADER_STAGE_COMPUTE_BIT);
      input.color = builder.sample_image(color, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(target, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(pipeline, 0);
      auto blk = cmd.allocate_ubo<TAAParams>();
      *blk.ptr = consts;

      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.history_color), sampler},
        gpu::TextureBinding {1, resources.get_view(input.history_depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.current_depth), sampler},
        gpu::TextureBinding {3, resources.get_view(input.velocity), sampler},
        gpu::TextureBinding {4, resources.get_view(input.color), sampler},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)},
        gpu::UBOBinding {6, cmd.get_ubo_pool(), blk}
      );

      const auto &extent = resources.get_image(input.out)->get_extent();

      cmd.bind_pipeline(pipeline);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset});
      //cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch((extent.width + 7)/8, (extent.height + 7)/8, 1);
    });
}

void TAA::remap_targets(rendergraph::RenderGraph &graph) {
  graph.remap(history, target);
}