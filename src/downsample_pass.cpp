#include "downsample_pass.hpp"
#include "gpu/gpu.hpp"
DownsamplePass::DownsamplePass() {
  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  gpu::Registers regs {};
  regs.depth_stencil.depthTestEnable = VK_TRUE;
  regs.depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
  regs.depth_stencil.depthWriteEnable = VK_TRUE;

  downsample_gbuffer = gpu::create_graphics_pipeline();
  downsample_gbuffer.set_program("downsample_gbuffer");
  downsample_gbuffer.set_registers(regs);
  downsample_gbuffer.set_vertex_input({});
  downsample_gbuffer.set_rendersubpass({true, {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT}});

  downsample_depth = gpu::create_graphics_pipeline();
  downsample_depth.set_program("depth_mips");
  downsample_depth.set_registers(regs);
  downsample_depth.set_vertex_input({});
  downsample_depth.set_rendersubpass({true, {VK_FORMAT_D24_UNORM_S8_UINT}});
  //downsample_gbuffer.set_rendersubpass({true, {desc.format}}); 
}

void DownsamplePass::run_downsample_gbuff(
  rendergraph::RenderGraph &graph,
  rendergraph::ImageResourceId src_normals,
  rendergraph::ImageResourceId src_velocity,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId out_normal,
  rendergraph::ImageResourceId out_velocity)
{
  auto depth_desc = graph.get_descriptor(depth);
  auto norm_desc = graph.get_descriptor(out_normal);
  auto vel_desc = graph.get_descriptor(out_velocity);

  if (depth_desc.mip_levels < 2) {
    throw std::runtime_error {"Can't downsample depth texture with 1 mip level"};
  }

  auto depth_ext = depth_desc.extent2D();
  auto norm_ext = norm_desc.extent2D();
  auto vel_ext = vel_desc.extent2D();

  depth_ext.width = std::max(1u, depth_ext.width/2);
  depth_ext.height = std::max(1u, depth_ext.height/2);

  if (depth_ext.width != norm_ext.width || depth_ext.height != norm_ext.height || vel_ext.width != norm_ext.width || vel_ext.height != norm_ext.height) {
    throw std::runtime_error {"Output textures have different sizes"};
  }

  downsample_gbuffer.set_rendersubpass({true, {norm_desc.format, vel_desc.format, depth_desc.format}});

  struct Input {
    rendergraph::ImageViewId gbuffer_depth;
    rendergraph::ImageViewId gbuffer_normal;
    rendergraph::ImageViewId gbuffer_velocity;
    rendergraph::ImageViewId out_depth;
    rendergraph::ImageViewId out_normal;
    rendergraph::ImageViewId out_velocity;
  };
  
  graph.add_task<Input>("DownsampleGbuffer",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder){
      input.gbuffer_depth = builder.sample_image(depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.gbuffer_normal = builder.sample_image(src_normals, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
      input.gbuffer_velocity = builder.sample_image(src_velocity, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
      input.out_depth = builder.use_depth_attachment(depth, 1, 0);
      input.out_normal = builder.use_color_attachment(out_normal, 0, 0);
      input.out_velocity = builder.use_color_attachment(out_velocity, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
       auto set = resources.allocate_set(downsample_gbuffer, 0); 
        gpu::write_set(set, 
          gpu::TextureBinding {0, resources.get_view(input.gbuffer_depth), sampler},
          gpu::TextureBinding {1, resources.get_view(input.gbuffer_normal), sampler},
          gpu::TextureBinding {2, resources.get_view(input.gbuffer_velocity), sampler});

        cmd.set_framebuffer(norm_ext.width, norm_ext.height, {
          resources.get_image_range(input.out_normal),
          resources.get_image_range(input.out_velocity),
          resources.get_image_range(input.out_depth)});

        cmd.bind_pipeline(downsample_gbuffer);
        cmd.bind_descriptors_graphics(0, {set});
        cmd.bind_viewport(0.f, 0.f, float(norm_ext.width), float(norm_ext.height), 0.f, 1.f);
        cmd.bind_scissors(0, 0, norm_ext.width, norm_ext.height);
        cmd.draw(3, 1, 0, 0);
        cmd.end_renderpass();
    });
  
}

void DownsamplePass::run_downsample_depth(
  rendergraph::RenderGraph &graph,
  rendergraph::ImageResourceId depth,
  uint32_t src_mip) //mip that will be readed from
{
  auto desc = graph.get_descriptor(depth);
  downsample_depth.set_rendersubpass({true, {desc.format}});

  struct Input {
    rendergraph::ImageViewId depth_tex;
    rendergraph::ImageViewId depth_rt;
  };

  for (uint32_t i = src_mip + 1; i < desc.mip_levels; i++) {
    graph.add_task<Input>("DownsampleDepth",
      [&](Input &input, rendergraph::RenderGraphBuilder &builder){
        input.depth_rt = builder.use_depth_attachment(depth, i, 0);
        input.depth_tex = builder.sample_image(depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, i - 1, 1, 0, 1);
      },
      [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
        auto set = resources.allocate_set(downsample_depth, 0); 
        gpu::write_set(set, 
          gpu::TextureBinding {0, resources.get_view(input.depth_tex), sampler});

        uint32_t w = desc.width/(1 << i), h = desc.height/(1 << i);
        w = std::max(w, 1u);
        h = std::max(h, 1u);

        cmd.set_framebuffer(w, h, {resources.get_image_range(input.depth_rt)});
        cmd.bind_pipeline(downsample_depth);
        cmd.bind_descriptors_graphics(0, {set});
        cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
        cmd.bind_scissors(0, 0, w, h);
        cmd.draw(3, 1, 0, 0);
        cmd.end_renderpass();
      });
  }
}

void DownsamplePass::run(
  rendergraph::RenderGraph &graph,
  rendergraph::ImageResourceId src_normals,
  rendergraph::ImageResourceId src_velocity,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId out_normals,
  rendergraph::ImageResourceId out_velocity)
{
  run_downsample_gbuff(graph, src_normals, src_velocity, depth, out_normals, out_velocity);
  run_downsample_depth(graph, depth, 1);
}