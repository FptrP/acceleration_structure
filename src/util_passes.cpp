#include "util_passes.hpp"

#include "gpu/gpu.hpp"

void gen_perlin_noise2D(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, uint32_t mip, uint32_t layer) {
  struct Data {
    rendergraph::ImageViewId rt;
    gpu::GraphicsPipeline pipeline;  
  };
  
  gpu::GraphicsPipeline pipeline {gpu::create_graphics_pipeline()};
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  pipeline.set_program("perlin");

  graph.add_task<Data>("Perlin",
    [&](Data &data, rendergraph::RenderGraphBuilder &builder){
      data.rt = builder.use_color_attachment(image, mip, layer);

      auto fmt = builder.get_image_info(image).format;
      data.pipeline = pipeline;
      data.pipeline.set_rendersubpass({false, {fmt}});
    },
    [=](Data &data, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      const auto &image_desc = resources.get_image(data.rt)->get_extent();

      cmd.set_framebuffer(image_desc.width, image_desc.height, {resources.get_image_range(data.rt)});
      cmd.bind_pipeline(data.pipeline);
      cmd.bind_viewport(0.f, 0.f, float(image_desc.width), float(image_desc.height), 0.f, 1.f);
      cmd.bind_scissors(0, 0, image_desc.width, image_desc.height);
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    });

}


void gen_mipmaps(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image) {
  struct Data {};

  auto desc = graph.get_descriptor(image); 

  for (uint32_t dst_mip = 1; dst_mip < desc.mip_levels; dst_mip++) {
    uint32_t src_mip = dst_mip - 1;

    graph.add_task<Data>("Genmips",
      [&](Data &, rendergraph::RenderGraphBuilder &builder){
        builder.transfer_read(image, src_mip, 1, 0, 1);
        builder.transfer_write(image, dst_mip, 1, 0, 1);
      },
      [=](Data &, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
        auto img_ptr = resources.get_image(image);
        const auto &ext = img_ptr->get_extent(); 
        auto api_image = img_ptr->api_image();
        auto aspect = img_ptr->get_default_aspect();

        int32_t src_width = ext.width/(1 << src_mip);
        int32_t src_height = ext.height/(1 << src_mip);
        
        VkImageBlit region {};
        region.srcOffsets[0] = VkOffset3D {0, 0, 0};
        region.srcOffsets[1] = VkOffset3D {src_width, src_height, 1};
        region.dstOffsets[0] = VkOffset3D {0, 0, 0};
        region.dstOffsets[1] = VkOffset3D {src_width/2, src_height/2, 1};
        region.srcSubresource = VkImageSubresourceLayers {
          .aspectMask = uint32_t(aspect),
          .mipLevel = src_mip,
          .baseArrayLayer = 0,
          .layerCount = 1
        };

        region.dstSubresource = VkImageSubresourceLayers {
          .aspectMask = uint32_t(aspect),
          .mipLevel = dst_mip,
          .baseArrayLayer = 0,
          .layerCount = 1
        }; 

        vkCmdBlitImage(
          cmd.get_command_buffer(),
          api_image,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          api_image,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          1,
          &region,
          VK_FILTER_LINEAR);
      });
  }

}

void clear_depth(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, float val) {
  struct Data {};

  auto info = graph.get_descriptor(image);

  graph.add_task<Data>("Clear_depth",
    [&](Data &, rendergraph::RenderGraphBuilder &builder){
      builder.transfer_write(image, 0, info.mip_levels, 0, info.array_layers);
    },
    [=](Data &, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      VkClearDepthStencilValue clear_val {.depth = val};
      VkImageSubresourceRange range {
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0,
        info.mip_levels,
        0,
        info.array_layers
      };

      vkCmdClearDepthStencilImage(
        cmd.get_command_buffer(),
        resources.get_image(image)->api_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        &clear_val,
        1,
        &range
      );
    });
}

void clear_color(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, VkClearColorValue val) {
  struct Data {};

  auto info = graph.get_descriptor(image);

  graph.add_task<Data>("Clear_color",
    [&](Data &, rendergraph::RenderGraphBuilder &builder){
      builder.transfer_write(image, 0, info.mip_levels, 0, info.array_layers);
    },
    [=](Data &, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){

      VkImageSubresourceRange range {
        VK_IMAGE_ASPECT_COLOR_BIT,
        0,
        info.mip_levels,
        0,
        info.array_layers
      };

      vkCmdClearColorImage(
        cmd.get_command_buffer(),
        resources.get_image(image)->api_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        &val,
        1,
        &range);
    });  
}

void blit_image(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId src, rendergraph::ImageResourceId dst) {
  struct Input {};
  graph.add_task<Input>("CopyImage",
    [&](Input &, rendergraph::RenderGraphBuilder &builder){
      builder.transfer_read(src, 0, 1, 0, 1);
      builder.transfer_write(dst, 0, 1, 0, 1);
    },
    [=](Input &, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto src_ext = resources.get_image(src)->get_extent();
      auto dst_ext = resources.get_image(dst)->get_extent();
      VkImageBlit region {
        .srcSubresource {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets {{0, 0, 0}, {(int32_t)src_ext.width, (int32_t)src_ext.height, 1}},
        .dstSubresource {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffsets {{0, 0, 0}, {(int32_t)dst_ext.width, (int32_t)dst_ext.height, 1}}
      };

      vkCmdBlitImage(
        cmd.get_command_buffer(),
        resources.get_image(src)->api_image(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        resources.get_image(dst)->api_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, 
        &region,
        VK_FILTER_LINEAR);
    
    });
}