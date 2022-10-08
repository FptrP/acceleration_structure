#include "backbuffer_subpass2.hpp"

#include <memory>
#include <vector>
#include "imgui_pass.hpp"

struct SubpassData {
  rendergraph::ImageViewId backbuff_view;
  rendergraph::ImageViewId texture_view;
};

static gpu::GraphicsPipeline pipeline;
struct Nil {};

void add_backbuffer_subpass(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId draw_img, VkSampler sampler, DrawTex flags) {
  
  pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("texdraw");
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  
  graph.add_task<SubpassData>("BackbufSubpass",
    [&](SubpassData &data, rendergraph::RenderGraphBuilder &builder){
      data.backbuff_view = builder.use_backbuffer_attachment();
      data.texture_view = builder.sample_image(draw_img, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 0, 1, 0, 1);

      auto desc = builder.get_image_info(data.backbuff_view);
      pipeline.set_rendersubpass({false, {desc.format}});
    },
    [=](SubpassData &data, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      const auto &ext = resources.get_image(data.backbuff_view)->get_extent();
      
      auto set = resources.allocate_set(pipeline.get_layout(0));

      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(data.texture_view), sampler});

      VkRect2D scissors {{0, 0}, VkExtent2D {ext.width, ext.height}};
      VkViewport viewport {0.f, 0.f, (float)ext.width, (float)ext.height, 0.f, 1.f};

      cmd.set_framebuffer(ext.width, ext.height, {resources.get_image_range(data.backbuff_view)});
      cmd.bind_pipeline(pipeline);
      cmd.clear_color_attachments(0.f, 0.f, 0.f, 0.f);
      cmd.bind_descriptors_graphics(0, {set}, {});
      cmd.push_constants_graphics(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(flags), &flags);
      cmd.bind_viewport(viewport);
      cmd.bind_scissors(scissors);
      cmd.draw(3, 1, 0, 0);
      imgui_draw(cmd.get_command_buffer());
      cmd.end_renderpass();
    });
}

void add_backbuffer_subpass(rendergraph::RenderGraph &graph, gpu::ImagePtr &image, VkSampler sampler, DrawTex flags) {
  pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("texdraw");
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  
  struct SubpassData {
    rendergraph::ImageViewId backbuff_view;
  };

  graph.add_task<SubpassData>("BackbufSubpass",
    [&](SubpassData &data, rendergraph::RenderGraphBuilder &builder){
      data.backbuff_view = builder.use_backbuffer_attachment();
      auto desc = builder.get_image_info(data.backbuff_view);
      pipeline.set_rendersubpass({false, {desc.format}});
    },
    [=](SubpassData &data, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) mutable {
      const auto &ext = resources.get_image(data.backbuff_view)->get_extent();
      
      auto set = resources.allocate_set(pipeline.get_layout(0));
      auto range = gpu::make_image_range2D(0, ~0u);

      gpu::write_set(set,
        gpu::TextureBinding {0, image->get_view(range), sampler});

      VkRect2D scissors {{0, 0}, VkExtent2D {ext.width, ext.height}};
      VkViewport viewport {0.f, 0.f, (float)ext.width, (float)ext.height, 0.f, 1.f};

      cmd.set_framebuffer(ext.width, ext.height, {resources.get_image_range(data.backbuff_view)});
      cmd.bind_pipeline(pipeline);
      cmd.clear_color_attachments(0.f, 0.f, 0.f, 0.f);
      cmd.bind_descriptors_graphics(0, {set}, {});
      cmd.push_constants_graphics(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(flags), &flags);
      cmd.bind_viewport(viewport);
      cmd.bind_scissors(scissors);
      cmd.draw(3, 1, 0, 0);
      imgui_draw(cmd.get_command_buffer());
      cmd.end_renderpass();
    });
}

void add_present_subpass(rendergraph::RenderGraph &graph) {
  struct Nil {};
  graph.add_task<Nil>("presentPrepare",
  [&](Nil &, rendergraph::RenderGraphBuilder &builder){
    builder.prepare_backbuffer();
  },
  [=](Nil &, rendergraph::RenderResources&, gpu::CmdContext &){

  });
}