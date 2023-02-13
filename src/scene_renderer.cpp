#include "scene_renderer.hpp"
#include "gpu_transfer.hpp"
#include "gpu/imgui_context.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>

Gbuffer::Gbuffer(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) : w {width}, h {height} {
  auto tiling = VK_IMAGE_TILING_OPTIMAL;
  auto color_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  auto depth_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  auto triangle_id_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  uint32_t depth_mips = std::floor(std::log2(std::max(width, height))) + 1;

  gpu::ImageInfo albedo_info {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo normal_info {VK_FORMAT_R16G16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo velocity_info {VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo mat_info {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo triangle_id_info {VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, width, height};

  gpu::ImageInfo depth_info {
    VK_FORMAT_D24_UNORM_S8_UINT, 
    VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT,
    width,
    height,
    1,
    depth_mips,
    1
  };

  albedo = graph.create_image(VK_IMAGE_TYPE_2D, albedo_info, tiling, VK_IMAGE_USAGE_STORAGE_BIT|color_usage);
  normal = graph.create_image(VK_IMAGE_TYPE_2D, normal_info, tiling, VK_IMAGE_USAGE_STORAGE_BIT|color_usage);
  velocity_vectors = graph.create_image(VK_IMAGE_TYPE_2D, velocity_info, tiling, VK_IMAGE_USAGE_STORAGE_BIT|color_usage);
  triangle_id = graph.create_image(VK_IMAGE_TYPE_2D, triangle_id_info, tiling, triangle_id_flags);

  normal_info.width /= 2;
  normal_info.height /= 2;
  velocity_info.width /= 2;
  velocity_info.height /= 2;

  downsampled_normals = graph.create_image(VK_IMAGE_TYPE_2D, normal_info, tiling, color_usage);
  downsampled_velocity_vectors = graph.create_image(VK_IMAGE_TYPE_2D, velocity_info, tiling, color_usage);
  
  material = graph.create_image(VK_IMAGE_TYPE_2D, mat_info, tiling, VK_IMAGE_USAGE_STORAGE_BIT|color_usage);
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

LightsManager::LightsManager(rendergraph::RenderGraph &graph)
{
  lights_buf = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(Light) * MAX_LIGHTS, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  
  lights_pipeline = gpu::create_graphics_pipeline();
  lights_pipeline.set_program("draw_lights");
  lights_pipeline.set_vertex_input({});
  
  gpu::Registers regs {};
  regs.depth_stencil.depthTestEnable = VK_TRUE;
  regs.depth_stencil.depthWriteEnable = VK_FALSE;
  lights_pipeline.set_registers(regs);
}

static inline bool modify_set(float &dst, float src)
{
  float t = dst;
  dst = src;
  return fabs(t - src) > 1e-6;
}

static inline bool modify_set(glm::vec4 &dst, const glm::vec4 &src)
{
  bool modified = false;
  modified |= modify_set(dst.x, src.x);
  modified |= modify_set(dst.y, src.y);
  modified |= modify_set(dst.z, src.z);
  modified |= modify_set(dst.w, src.w);
  return modified;
}

void LightsManager::set(uint32_t i, glm::vec3 pos, glm::vec3 color)
{
  auto &light = lights.at(i);

  glm::vec4 lpos {pos, 1.f}; 
  glm::vec4 lcol {color, 1.f};

  light_changed |= modify_set(light.pos, lpos);
  light_changed |= modify_set(light.color, lcol);
}

void LightsManager::update() {
  gpu_transfer::write_buffer(lights_buf, 0, sizeof(Light) * lights.size(), lights.data());
}

void LightsManager::update_imgui() {
  ImGui::Begin("Lights");
  
  for (uint32_t i = 0; i < lights.size(); i++) {
    char name[32];
    sprintf(name, "position%d", i);
    light_changed |= ImGui::InputFloat3(name, &lights[i].pos.x);
    sprintf(name, "color%d", i);
    light_changed |= ImGui::SliderFloat3(name, &lights[i].color.x, 0.f, 5.f);
  }
  
  ImGui::End();
}

void LightsManager::draw_lights(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId target, rendergraph::ImageResourceId depth, const glm::mat4 &proj, const DrawTAAParams &params) {
  struct PushConstants {
    glm::mat4 projection;
    glm::vec4 position;
    glm::vec4 color;
  };

  struct Input {
    rendergraph::ImageViewId rt;
    rendergraph::ImageViewId depth;
  };

  glm::mat4 camera = params.camera; 

  graph.add_task<Input>("DrawLights",
  [&](Input &input, rendergraph::RenderGraphBuilder &builder){
    input.rt = builder.use_color_attachment(target, 0, 0);
    input.depth = builder.use_depth_attachment(depth, 0, 0);
    
    lights_pipeline.set_rendersubpass({true, {
      builder.get_image_info(target).format,
      builder.get_image_info(depth).format}});
  },
  [=](Input &input, rendergraph::RenderResources &res, gpu::CmdContext &cmd){
    auto ext = res.get_image(input.rt)->get_extent();
    
    cmd.set_framebuffer(ext.width, ext.height, {
      res.get_image_range(input.rt),
      res.get_image_range(input.depth)  
    });
    
    cmd.bind_pipeline(lights_pipeline);
    cmd.bind_viewport(0.f, 0.f, ext.width, ext.height, 0.f, 1.f);
    cmd.bind_scissors(0, 0, ext.width, ext.height);

    for (uint32_t i = 0; i < lights.size(); i++) {
      const auto &light = lights[i];
      PushConstants pc {};
      auto pos = light.pos;
      pos.w = 1.f;
      pc.position = camera * pos;
      pc.color = light.color;
      pc.projection = proj;
      cmd.push_constants_graphics(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
      cmd.draw(6, 1, 0, 0);
    }
  });
  

}

void SceneRenderer::init_pipeline(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer) {
  gpu::Registers regs {};
  regs.depth_stencil.depthTestEnable = VK_TRUE;
  regs.depth_stencil.depthWriteEnable = VK_TRUE;

  opaque_taa_pipeline = gpu::create_graphics_pipeline();
  opaque_taa_pipeline.set_program("gbuf_opaque_taa");
  opaque_taa_pipeline.set_registers(regs);
  opaque_taa_pipeline.set_vertex_input(scene::get_vertex_input());    
  opaque_taa_pipeline.set_rendersubpass({true, {
    graph.get_descriptor(gbuffer.albedo).format, 
    VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R16G16_SFLOAT,
    VK_FORMAT_D24_UNORM_S8_UINT
  }});

  triangle_id_pipeline = gpu::create_graphics_pipeline();
  triangle_id_pipeline.set_program("gbuf_triangle_id");
  triangle_id_pipeline.set_registers(regs);
  triangle_id_pipeline.set_vertex_input(scene::get_vertex_input_shadow());
  triangle_id_pipeline.set_rendersubpass({true, {
    VK_FORMAT_R32_UINT,
    VK_FORMAT_D24_UNORM_S8_UINT
  }});

  reconstruct_pipeline = gpu::create_compute_pipeline("gbuf_reconstruct");

  auto sampler_info = gpu::DEFAULT_SAMPLER;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;

  sampler = gpu::create_sampler(sampler_info);
  
  sampler_info.minFilter = VK_FILTER_NEAREST;
  sampler_info.magFilter = VK_FILTER_NEAREST;
  integer_sampler = gpu::create_sampler(sampler_info);
  
  transform_buffer = graph.create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(glm::mat4) * 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  drawcall_buffer = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(DrawCall) * MAX_DRAWCALLS, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

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

static void node_process(const scene::CompiledScene &target, const scene::BaseNode &node, std::vector<SceneRenderer::DrawCall> &draw_calls, std::vector<glm::mat4> &transforms, const glm::mat4 &acc) {
  auto transform = acc * node.transform;
  
  if (node.mesh_index >= 0) {
    assert(node.transform_index >= 0);
    transforms[2 * node.transform_index] = transform;
    transforms[2 * node.transform_index + 1] = glm::transpose(glm::inverse(transform));

    for (auto prim_id : target.root_meshes[node.mesh_index].primitive_indexes)
      draw_calls.push_back(SceneRenderer::DrawCall {(uint32_t)node.transform_index, prim_id});
  }

  for (auto &child : node.children) {
    node_process(target, child, draw_calls, transforms, transform);
  }
}

void SceneRenderer::update_scene() {
  auto identity = glm::identity<glm::mat4>();
  std::vector<glm::mat4> transforms;

  transforms.resize(2 * target.transforms_count);
  draw_calls.clear();
  
  for (auto &node : target.base_nodes) {
    node_process(target, node, draw_calls, transforms, identity);
  }

  gpu_transfer::write_buffer(transform_buffer, 0, sizeof(glm::mat4) * transforms.size(), transforms.data());
  gpu_transfer::write_buffer(drawcall_buffer, 0, sizeof(DrawCall) * draw_calls.size(), draw_calls.data());
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
        const auto &prim = target.primitives[draw_call.primitive];
        const auto &material = target.materials[prim.material_index];

        PushData pc {};
        pc.transform_index = draw_call.transform;
        pc.albedo_index = (material.albedo_tex_index < scene_textures.size())? material.albedo_tex_index : scene::INVALID_TEXTURE;
        pc.mr_index = (material.metalic_roughness_index < scene_textures.size())? material.metalic_roughness_index : scene::INVALID_TEXTURE;
        pc.flags = material.clip_alpha? 0xff : 0;
        
        cmd.push_constants_graphics(VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushData), &pc);
        cmd.draw_indexed(prim.index_count, 1, prim.index_offset, prim.vertex_offset, 0);
      }

      cmd.end_renderpass();
      
    });
}

void SceneRenderer::rasterize_triange_id(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const DrawTAAParams &params) {
  struct Data {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId triangle_id;
  };
  
  struct PushData {
    uint32_t transform_index;
    uint32_t drawcall_index;
  };

  struct GbufConst {
    glm::mat4 view_projection;
    glm::vec4 jitter;
  };
  
  GbufConst consts {params.mvp, params.jitter};

  graph.add_task<Data>("TriangleIdPass",
    [&](Data &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.use_depth_attachment(gbuffer.depth, 0, 0);
      input.triangle_id = builder.use_color_attachment(gbuffer.triangle_id, 0, 0);
      builder.use_storage_buffer(transform_buffer, VK_SHADER_STAGE_VERTEX_BIT);
    },
    [=](Data &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      cmd.set_framebuffer(gbuffer.w, gbuffer.h, {
        resources.get_image_range(input.triangle_id),
        resources.get_image_range(input.depth)
      });
      
      cmd.bind_pipeline(triangle_id_pipeline);
      cmd.clear_color_attachments(~0u, ~0u, ~0u, ~0u);
      cmd.clear_depth_attachment(1.f);
      cmd.bind_viewport(0.f, 0.f, gbuffer.w, gbuffer.h, 0.f, 1.f);
      cmd.bind_scissors(0, 0, gbuffer.w, gbuffer.h);
      
      auto blk = cmd.allocate_ubo<GbufConst>();
      *blk.ptr = consts;

      auto set = resources.allocate_set(triangle_id_pipeline, 0);

      gpu::write_set(set, 
        gpu::SSBOBinding {0, resources.get_buffer(transform_buffer)},
        gpu::UBOBinding {1, cmd.get_ubo_pool(), blk});

      cmd.bind_descriptors_graphics(0, {set}, {blk.offset});
      cmd.bind_vertex_buffers(0, {target.vertex_buffer->api_buffer()}, {0ul});
      cmd.bind_index_buffer(target.index_buffer->api_buffer(), 0, VK_INDEX_TYPE_UINT32);

      uint32_t drawcall_id = 0;

      for (const auto &draw_call : draw_calls) {
        const auto &prim = target.primitives[draw_call.primitive];

        PushData pc {};
        pc.transform_index = draw_call.transform;
        pc.drawcall_index = drawcall_id;
        
        cmd.push_constants_graphics(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushData), &pc);
        cmd.draw_indexed(prim.index_count, 1, prim.index_offset, prim.vertex_offset, 0);
        drawcall_id++;
      }

      cmd.end_renderpass();
      
    });

}

void SceneRenderer::reconstruct_gbuffer(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const DrawTAAParams &params) {
  struct Data {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId triangle_id;
    
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId velocity;
  };
  
  struct UniformConst {
    glm::mat4 camera;
    glm::mat4 view_projection;
    glm::mat4 prev_view_projection;
    glm::vec4 jitter;
    glm::vec4 fovy_aspect_znear_zfar;
  };
  
  UniformConst consts {params.camera, params.mvp, params.prev_mvp, params.jitter, params.fovy_aspect_znear_zfar};

  graph.add_task<Data>("GbufferReconstructPass",
    [&](Data &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(gbuffer.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
      input.triangle_id = builder.sample_image(gbuffer.triangle_id, VK_SHADER_STAGE_COMPUTE_BIT);
      
      input.albedo = builder.use_storage_image(gbuffer.albedo, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      input.normal = builder.use_storage_image(gbuffer.normal, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      input.material = builder.use_storage_image(gbuffer.material, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      input.velocity = builder.use_storage_image(gbuffer.velocity_vectors, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);

      builder.use_storage_buffer(transform_buffer, VK_SHADER_STAGE_COMPUTE_BIT);
      builder.use_storage_buffer(drawcall_buffer, VK_SHADER_STAGE_COMPUTE_BIT);
    },
    [=](Data &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto blk = cmd.allocate_ubo<UniformConst>(); 
      *blk.ptr = consts;

      auto set = resources.allocate_set(reconstruct_pipeline, 0);
      gpu::write_set(set, 
        gpu::UBOBinding {0, cmd.get_ubo_pool(), blk},
        gpu::SSBOBinding {1, resources.get_buffer(transform_buffer)},
        gpu::SSBOBinding {2, target.vertex_buffer},
        gpu::SSBOBinding {3, target.index_buffer},
        gpu::SSBOBinding {4, target.material_buffer},
        gpu::SSBOBinding {5, target.primitive_buffer},
        gpu::TextureBinding {6, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {7, resources.get_view(input.triangle_id), integer_sampler},
        gpu::StorageTextureBinding {8, resources.get_view(input.albedo)},
        gpu::StorageTextureBinding {9, resources.get_view(input.normal)},
        gpu::StorageTextureBinding {10, resources.get_view(input.material)},
        gpu::StorageTextureBinding {11, resources.get_view(input.velocity)},
        gpu::SSBOBinding {12, resources.get_buffer(drawcall_buffer)}
      );

      auto extent = resources.get_image(input.albedo)->get_extent();

      cmd.bind_pipeline(reconstruct_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset});
      cmd.bind_descriptors_compute(1, {bindless_textures}, {});
      cmd.dispatch((extent.width + 7)/8, (extent.height + 3)/4, 1);
    });
}