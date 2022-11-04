#include "rtfx.hpp"

RTReflections::RTReflections(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo rays_info {VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  result = graph.create_image(VK_IMAGE_TYPE_2D, rays_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);
  pipeline = gpu::create_compute_pipeline("trace_depth_as");
  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
}
  
void RTReflections::run(rendergraph::RenderGraph &graph, const Gbuffer &gbuff, const DepthAs &depth_as, const AdvancedSSRParams &params) {

  struct ShaderParams {
    glm::mat4 normal_mat;
    uint32_t frame_random;
    float fovy;
    float aspect;
    float znear;
    float zfar;
  };

  ShaderParams ubo_data {
    params.normal_mat,
    0,
    params.fovy,
    params.aspect,
    params.znear,
    params.zfar
  };
  
  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId albedo;
    VkAccelerationStructureKHR acstruct;
    rendergraph::ImageViewId out;
  };
  
  
  graph.add_task<Input>("RTReflections",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.depth = builder.sample_image(gbuff.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1, 0, 1);
      input.normal = builder.sample_image(gbuff.downsampled_normals, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(result, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      input.albedo = builder.sample_image(gbuff.albedo, VK_SHADER_STAGE_COMPUTE_BIT);
      input.acstruct = depth_as.get_tlas();
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(pipeline, 0);
      auto blk = cmd.allocate_ubo<ShaderParams>();
      *blk.ptr = ubo_data;

      gpu::write_set(set, 
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.albedo), sampler},
        gpu::UBOBinding {3, cmd.get_ubo_pool(), blk},
        gpu::AccelerationStructBinding {4, input.acstruct},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)});
      
      auto ext = resources.get_image(input.out)->get_extent();
      cmd.bind_pipeline(pipeline);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset});
      //cmd.push_constants_compute(0, sizeof(push_consts), &push_consts);
      cmd.dispatch((ext.width + 7)/8, (ext.height + 3)/4, 1);
    });

}