#include "ssr.hpp"

#include "gpu/gpu.hpp"

rendergraph::ImageResourceId create_ssr_tex(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h) {
  gpu::ImageInfo info {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
  return graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);
}

void add_ssr_pass(
  rendergraph::RenderGraph &graph,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId normal,
  rendergraph::ImageResourceId color,
  rendergraph::ImageResourceId material,
  rendergraph::ImageResourceId out,
  const SSRParams &params)
{
  auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER); 
  
  auto depth_sampler_info = gpu::DEFAULT_SAMPLER;
  depth_sampler_info.minFilter = VK_FILTER_NEAREST;
  depth_sampler_info.magFilter = VK_FILTER_NEAREST;
  depth_sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  depth_sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  depth_sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

  auto depth_sampler = gpu::create_sampler(depth_sampler_info);

  auto pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("ssr");
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  pipeline.set_rendersubpass({false, {graph.get_descriptor(out).format}});

  struct Input {
    rendergraph::ImageViewId depth, normal, color, material, rt;
  };

  graph.add_task<Input>("SSR",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
      input.normal = builder.sample_image(normal, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.color = builder.sample_image(color, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.material = builder.sample_image(material, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.rt = builder.use_color_attachment(out, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      
      auto block = cmd.allocate_ubo<SSRParams>();
      *block.ptr = params;

      auto set = resources.allocate_set(pipeline, 0);

      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {1, resources.get_view(input.depth), depth_sampler},
        gpu::TextureBinding {2, resources.get_view(input.color), sampler},
        gpu::UBOBinding {3, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {4, resources.get_view(input.material), sampler});

      const auto &image_info = resources.get_image(input.rt)->get_extent();
      auto w = image_info.width;
      auto h = image_info.height;

      cmd.set_framebuffer(w, h, {resources.get_image_range(input.rt)});
      cmd.bind_pipeline(pipeline);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_descriptors_graphics(0, {set}, {block.offset});
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    });
}