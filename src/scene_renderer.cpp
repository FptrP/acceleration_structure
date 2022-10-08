#include "scene_renderer.hpp"
#include "gpu_transfer.hpp"

#include <cstdlib>
#include <iostream>
#include <cmath>

Gbuffer::Gbuffer(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) : w {width}, h {height} {
  auto tiling = VK_IMAGE_TILING_OPTIMAL;
  auto color_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  auto depth_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  uint32_t depth_mips = std::floor(std::log2(std::max(width, height))) + 1;

  gpu::ImageInfo albedo_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo normal_info {VK_FORMAT_R16G16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo velocity_info {VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo mat_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo depth_info {
    VK_FORMAT_D24_UNORM_S8_UINT, 
    VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT,
    width,
    height,
    1,
    depth_mips,
    1
  };

  albedo = graph.create_image(VK_IMAGE_TYPE_2D, albedo_info, tiling, color_usage);
  normal = graph.create_image(VK_IMAGE_TYPE_2D, normal_info, tiling, color_usage);
  velocity_vectors = graph.create_image(VK_IMAGE_TYPE_2D, velocity_info, tiling, color_usage);

  normal_info.width /= 2;
  normal_info.height /= 2;
  velocity_info.width /= 2;
  velocity_info.height /= 2;

  downsampled_normals = graph.create_image(VK_IMAGE_TYPE_2D, normal_info, tiling, color_usage);
  downsampled_velocity_vectors = graph.create_image(VK_IMAGE_TYPE_2D, velocity_info, tiling, color_usage);
  
  material = graph.create_image(VK_IMAGE_TYPE_2D, mat_info, tiling, color_usage);
  depth = graph.create_image(VK_IMAGE_TYPE_2D, depth_info, tiling, depth_usage);
  prev_depth = graph.create_image(VK_IMAGE_TYPE_2D, depth_info, tiling, depth_usage|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

struct GbufConst {
  glm::mat4 camera;
  glm::mat4 projection;
  float fovy;
  float aspect;
  float z_near;
  float z_far; 
};

void SceneRenderer::init_pipeline(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer) {
  gpu::Registers regs {};
  regs.depth_stencil.depthTestEnable = VK_TRUE;
  regs.depth_stencil.depthWriteEnable = VK_TRUE;

  opaque_taa_pipeline = gpu::create_graphics_pipeline();
  opaque_taa_pipeline.set_program("gbuf_opaque_taa");
  opaque_taa_pipeline.set_registers(regs);
  opaque_taa_pipeline.set_vertex_input(scene::get_vertex_input());    
  opaque_taa_pipeline.set_rendersubpass({true, {
    VK_FORMAT_R8G8B8A8_SRGB, 
    VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_R16G16_SFLOAT,
    VK_FORMAT_D24_UNORM_S8_UINT
  }});

  shadow_pipeline = gpu::create_graphics_pipeline();
  shadow_pipeline.set_program("default_shadow");
  shadow_pipeline.set_registers(regs);
  shadow_pipeline.set_vertex_input(scene::get_vertex_input_shadow());

  auto sampler_info = gpu::DEFAULT_SAMPLER;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;

  sampler = gpu::create_sampler(sampler_info);
  transform_buffer = graph.create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(glm::mat4) * 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  scene_textures.reserve(target.textures.size());
  for (auto tex_desc : target.textures) {
    gpu::ImageViewRange range {VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1};
    auto &img = target.images[tex_desc.image_index];
    range.mips_count = img->get_mip_levels();

    scene_textures.push_back({img->get_view(range), target.samplers[tex_desc.sampler_index]});
  }

  uint32_t count = (uint32_t)scene_textures.size();
  if (count == 0) {
    count = 1;
  }

  bindless_textures = gpu::allocate_descriptor_set(opaque_taa_pipeline.get_layout(1), {count});
  
  if (scene_textures.size()) {
    gpu::write_set(bindless_textures, 
      gpu::ArrayOfImagesBinding {0, scene_textures});
  }  
}

static void node_process(const scene::BaseNode &node, std::vector<SceneRenderer::DrawCall> &draw_calls, std::vector<glm::mat4> &transforms, const glm::mat4 &acc) {
  auto transform = acc * node.transform;
  uint32_t transform_id = transforms.size()/2;
  
  if (node.mesh_index >= 0) {
    transforms.push_back(transform);
    transforms.push_back(glm::transpose(glm::inverse(transform)));
    draw_calls.push_back(SceneRenderer::DrawCall {transform_id, (uint32_t)node.mesh_index});
  }

  for (auto &child : node.children) {
    node_process(child, draw_calls, transforms, transform);
  }
}

void SceneRenderer::update_scene() {
  auto identity = glm::identity<glm::mat4>();
  std::vector<glm::mat4> transforms;

  draw_calls.clear();
  
  for (auto &node : target.base_nodes) {
    node_process(node, draw_calls, transforms, identity);
  }
  gpu_transfer::write_buffer(transform_buffer, 0, sizeof(glm::mat4) * transforms.size(), transforms.data());
}

struct PushData {
  uint32_t transform_index;
  uint32_t albedo_index;
  uint32_t mr_index;
  uint32_t flags;
};

void SceneRenderer::draw_taa(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const DrawTAAParams &params) {
   struct Data {
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId velocity;
  };
  
  struct GbufConst {
    glm::mat4 view_projection;
    glm::mat4 prev_view_projection;
    glm::vec4 jitter;
    glm::vec4 fovy_aspect_znear_zfar;
  };
  
  GbufConst consts {params.mvp, params.prev_mvp, params.jitter, params.fovy_aspect_znear_zfar};

  graph.add_task<Data>("GbufferPass",
    [&](Data &input, rendergraph::RenderGraphBuilder &builder){
      input.albedo = builder.use_color_attachment(gbuffer.albedo, 0, 0);
      input.normal = builder.use_color_attachment(gbuffer.normal, 0, 0);
      input.material = builder.use_color_attachment(gbuffer.material, 0, 0);
      input.depth = builder.use_depth_attachment(gbuffer.depth, 0, 0);
      input.velocity = builder.use_color_attachment(gbuffer.velocity_vectors, 0, 0);

      builder.use_storage_buffer(transform_buffer, VK_SHADER_STAGE_VERTEX_BIT);
    },
    [=](Data &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      cmd.set_framebuffer(gbuffer.w, gbuffer.h, {
        resources.get_image_range(input.albedo),
        resources.get_image_range(input.normal),
        resources.get_image_range(input.material),
        resources.get_image_range(input.velocity),
        resources.get_image_range(input.depth)
      });

      auto vbuf = target.vertex_buffer->api_buffer();
      auto ibuf = target.index_buffer->api_buffer();
      
      cmd.bind_pipeline(opaque_taa_pipeline);
      cmd.clear_color_attachments(0.f, 0.f, 0.f, 0.f);
      cmd.clear_depth_attachment(1.f);
      cmd.bind_viewport(0.f, 0.f, gbuffer.w, gbuffer.h, 0.f, 1.f);
      cmd.bind_scissors(0, 0, gbuffer.w, gbuffer.h);
      cmd.bind_vertex_buffers(0, {vbuf}, {0ul});
      cmd.bind_index_buffer(ibuf, 0, VK_INDEX_TYPE_UINT32);
      
      auto blk = cmd.allocate_ubo<GbufConst>();
      *blk.ptr = consts;

      auto set = resources.allocate_set(opaque_taa_pipeline, 0);

      gpu::write_set(set, 
        gpu::UBOBinding {0, cmd.get_ubo_pool(), blk},
        gpu::SSBOBinding {1, resources.get_buffer(transform_buffer)});

      cmd.bind_descriptors_graphics(0, {set}, {blk.offset});
      cmd.bind_descriptors_graphics(1, {bindless_textures}, {});

      for (const auto &draw_call : draw_calls) {
        const auto &mesh = target.root_meshes[draw_call.mesh];
        
        for (auto &prim : mesh.primitives) {
          const auto &material = target.materials[prim.material_index];

          PushData pc {};
          pc.transform_index = draw_call.transform;
          pc.albedo_index = (material.albedo_tex_index < scene_textures.size())? material.albedo_tex_index : scene::INVALID_TEXTURE;
          pc.mr_index = (material.metalic_roughness_index < scene_textures.size())? material.metalic_roughness_index : scene::INVALID_TEXTURE;
          pc.flags = material.clip_alpha? 0xff : 0;
        
          cmd.push_constants_graphics(VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushData), &pc);
          cmd.draw_indexed(prim.index_count, 1, prim.index_offset, prim.vertex_offset, 0);
        }
      }

      cmd.end_renderpass();
      
    });
}

void SceneRenderer::render_shadow(rendergraph::RenderGraph &graph, const glm::mat4 &shadow_mvp, rendergraph::ImageResourceId out_tex, uint32_t layer) {
  struct Data {
    rendergraph::ImageViewId depth;
  };
  //TODO: new scene traverse
  /*graph.add_task<Data>("ShadowPass",
    [&](Data &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.use_depth_attachment(out_tex, 0, layer);
      builder.use_storage_buffer(transform_buffer, VK_SHADER_STAGE_VERTEX_BIT);
    },
    [=](Data &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto &depth_rt = resources.get_image(input.depth);
      auto w = depth_rt.get_info().width;
      auto h = depth_rt.get_info().width;

      shadow_pipeline.set_rendersubpass({true, {depth_rt.get_fmt()}});

      cmd.set_framebuffer(w, h, {resources.get_view(input.depth)});

      auto vbuf = target.vertex_buffer.get_api_buffer();
      auto ibuf = target.index_buffer.get_api_buffer();
      
      cmd.bind_pipeline(shadow_pipeline);
      cmd.clear_color_attachments(0.f, 0.f, 0.f, 0.f);
      cmd.clear_depth_attachment(1.f);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_vertex_buffers(0, {vbuf}, {0ul});
      cmd.bind_index_buffer(ibuf, 0, VK_INDEX_TYPE_UINT32);
      
      auto set = resources.allocate_set(shadow_pipeline.get_layout(0));
      auto block = cmd.allocate_ubo<glm::mat4>();
      *block.ptr = shadow_mvp;

      gpu::write_set(set, 
        gpu::UBOBinding {0, cmd.get_ubo_pool(), block},
        gpu::SSBOBinding {1, resources.get_buffer(transform_buffer)});

      cmd.bind_descriptors_graphics(0, {set}, {block.offset});

      for (const auto &draw_call : draw_calls) {
        const auto &mesh = target.meshes[draw_call.mesh];
        const auto &material = target.materials[mesh.material_index];

        cmd.push_constants_graphics(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(uint32_t), &draw_call.transform);
        cmd.draw_indexed(mesh.index_count, 1, mesh.index_offset, mesh.vertex_offset, 0);
      }


      cmd.end_renderpass();
      
    });*/
}