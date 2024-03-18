#include "defered_shading.hpp"
#include "gpu_transfer.hpp"

struct ShaderConstants {
  glm::mat4 inverse_camera;
  glm::mat4 camera;
  glm::mat4 shadow_mvp;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

DeferedShadingPass::DeferedShadingPass(rendergraph::RenderGraph &graph, SDL_Window *window) {
  auto format = graph.get_descriptor(graph.get_backbuffer()).format;
  
  pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("defered_shading");
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  pipeline.set_rendersubpass({false, {format}});

  imgui_init(window, pipeline.get_renderpass());

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  ubo_consts = graph.create_buffer(
    VMA_MEMORY_USAGE_GPU_ONLY, 
    sizeof(ShaderConstants), 
    VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

void DeferedShadingPass::update_params(const glm::mat4 &camera, const glm::mat4 &shadow, float fovy, float aspect, float znear, float zfar) {
  ShaderConstants consts {
    glm::inverse(camera),
    camera,
    shadow,
    fovy,
    aspect,
    znear,
    zfar 
  };

  gpu_transfer::write_buffer(ubo_consts, 0, sizeof(ShaderConstants), &consts);
}

void DeferedShadingPass::draw(rendergraph::RenderGraph &graph, 
  const Gbuffer &gbuffer,
  rendergraph::ImageResourceId shadow,
  rendergraph::ImageResourceId ssao,
  rendergraph::ImageResourceId brdf_tex,
  rendergraph::ImageResourceId reflections,
  LightsManager &lights,
  rendergraph::ImageResourceId out_image)
{
  struct PassData {
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId rt;
    rendergraph::ImageViewId shadow;
    rendergraph::ImageViewId ssao;
    rendergraph::ImageViewId ssr;
    rendergraph::ImageViewId brdf;
    rendergraph::BufferResourceId ubo;
  };

  struct PushConsts {
    glm::vec2 min_max_roughness;
    uint32_t show_ao;
  };
  PushConsts pc {min_max_roughness, only_ao? 1u : 0u};
  pipeline.set_rendersubpass({false, {graph.get_descriptor(out_image).format}});

  auto lights_buffer = lights.get_buffer();

  graph.add_task<PassData>("DeferedShading",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.albedo = builder.sample_image(gbuffer.albedo, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.normal = builder.sample_image(gbuffer.normal, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.material = builder.sample_image(gbuffer.material, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.depth = builder.sample_image(gbuffer.depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
      input.rt = builder.use_color_attachment(out_image, 0, 0);
      input.shadow = builder.sample_image(shadow, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 0, 1, 0, 1);
      input.ssao = builder.sample_image(ssao, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.ssr = builder.sample_image(reflections, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.brdf = builder.sample_image(brdf_tex, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.ubo = ubo_consts;
      builder.use_uniform_buffer(input.ubo, VK_SHADER_STAGE_FRAGMENT_BIT);
      builder.use_uniform_buffer(lights_buffer, VK_SHADER_STAGE_FRAGMENT_BIT);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      
      auto set = resources.allocate_set(pipeline, 0);
      
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.albedo), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.material), sampler},
        gpu::TextureBinding {3, resources.get_view(input.depth), sampler},
        gpu::UBOBinding {4, resources.get_buffer(input.ubo)}, 
        gpu::TextureBinding {5, resources.get_view(input.shadow), sampler},
        gpu::TextureBinding {6, resources.get_view(input.ssao), sampler},
        gpu::TextureBinding {7, resources.get_view(input.brdf), sampler},
        gpu::TextureBinding {8, resources.get_view(input.ssr), sampler},
        gpu::UBOBinding {9, resources.get_buffer(lights_buffer)});
      
      const auto &image_info = resources.get_image(input.rt)->get_extent();
      auto w = image_info.width;
      auto h = image_info.height;

      cmd.set_framebuffer(w, h, {resources.get_image_range(input.rt)});
      cmd.bind_pipeline(pipeline);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_descriptors_graphics(0, {set}, {0, 0});
      cmd.push_constants_graphics(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    }); 
}

void DeferedShadingPass::draw_ui() {
  ImGui::Begin("DeferedShading");
  ImGui::SliderFloat("Max Roughness", &min_max_roughness.y, min_max_roughness.x, 1.f);
  ImGui::SliderFloat("Min Roughness", &min_max_roughness.x, 0.f, min_max_roughness.y);
  ImGui::Checkbox("Show AO only", &only_ao);
  ImGui::End();
}


DiffuseSpecularPass::DiffuseSpecularPass(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo image_info {VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  
  diffuse_image = graph.create_image(VK_IMAGE_TYPE_2D, image_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);
  specular_image = graph.create_image(VK_IMAGE_TYPE_2D, image_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);
  
  pipeline = gpu::create_compute_pipeline("diffuse_specular_pass");
}
  
void DiffuseSpecularPass::run(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, rendergraph::ImageResourceId shadow,
                              rendergraph::ImageResourceId occlusion, const DrawTAAParams &params, LightsManager &lights)
{
  struct Input {
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId shadow;
    rendergraph::ImageViewId occlusion;
    rendergraph::ImageViewId out_diffuse;
    rendergraph::ImageViewId out_specular;
  };

  struct UBO {
    glm::mat4 inverse_camera;
    float fovy;
    float aspect;
    float znear;
    float zfar;
  };

  UBO ubo_data {glm::inverse(params.camera), params.fovy_aspect_znear_zfar.x, params.fovy_aspect_znear_zfar.y, params.fovy_aspect_znear_zfar.z, params.fovy_aspect_znear_zfar.w};
  auto lights_buffer = lights.get_buffer();

  graph.add_task<Input>("DiffuseSpecularPass",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.albedo = builder.sample_image(gbuffer.albedo, VK_SHADER_STAGE_COMPUTE_BIT);
      input.normal = builder.sample_image(gbuffer.normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(gbuffer.material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.depth = builder.sample_image(gbuffer.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
      input.shadow = builder.sample_image(shadow, VK_SHADER_STAGE_COMPUTE_BIT);
      input.occlusion = builder.sample_image(occlusion, VK_SHADER_STAGE_COMPUTE_BIT);

      builder.use_uniform_buffer(lights_buffer, VK_SHADER_STAGE_COMPUTE_BIT);

      input.out_diffuse = builder.use_storage_image(diffuse_image, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      input.out_specular = builder.use_storage_image(specular_image, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(pipeline, 0);
      auto blk = cmd.allocate_ubo<UBO>();
      *blk.ptr = ubo_data;

      auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

      gpu::write_set(set, 
        gpu::TextureBinding {0, resources.get_view(input.albedo), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.material), sampler},
        gpu::TextureBinding {3, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {4, resources.get_view(input.shadow), sampler},
        gpu::TextureBinding {5, resources.get_view(input.occlusion), sampler},
        gpu::UBOBinding {6, cmd.get_ubo_pool(), blk},
        gpu::UBOBinding {7, resources.get_buffer(lights_buffer)},
        gpu::StorageTextureBinding {8, resources.get_view(input.out_diffuse)},
        gpu::StorageTextureBinding {9, resources.get_view(input.out_specular)});
      
      auto ext = resources.get_image(input.out_diffuse)->get_extent();
      cmd.bind_pipeline(pipeline);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset, 0});
      //cmd.push_constants_compute(0, sizeof(push_consts), &push_consts);
      cmd.dispatch((ext.width + 7)/8, (ext.height + 3)/4, 1);
    });

}

LightResolvePass::LightResolvePass(rendergraph::RenderGraph &graph) {
  auto format = graph.get_descriptor(graph.get_backbuffer()).format;
  
  pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("light_resolve_pass");
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  pipeline.set_rendersubpass({false, {format}});
}

void LightResolvePass::run(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const DiffuseSpecularPass &diff_spec,
           rendergraph::ImageResourceId reflections, rendergraph::ImageResourceId final_image, const DrawTAAParams &params)
{
  pipeline.set_rendersubpass({false, {graph.get_descriptor(final_image).format}});

  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId diffuse;
    rendergraph::ImageViewId specular;
    rendergraph::ImageViewId reflections;
    rendergraph::ImageViewId result;
  };

  struct Constants {
    glm::mat4 normal_mat;
    float fovy, apsect, znear, zfar;
  };

  struct PushConstants {
    float reflectiveness;
  };

  Constants consts {glm::transpose(glm::inverse(params.camera)), params.fovy_aspect_znear_zfar.x, params.fovy_aspect_znear_zfar.y, params.fovy_aspect_znear_zfar.z, params.fovy_aspect_znear_zfar.w};
  PushConstants pc {reflectiveness};

  graph.add_task<Input>("LightResolve",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.normal = builder.sample_image(gbuffer.normal, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.material = builder.sample_image(gbuffer.material, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.depth = builder.sample_image(gbuffer.depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
      
      input.diffuse = builder.sample_image(diff_spec.get_diffuse(), VK_SHADER_STAGE_FRAGMENT_BIT);
      input.specular = builder.sample_image(diff_spec.get_specular(), VK_SHADER_STAGE_FRAGMENT_BIT);

      input.reflections = builder.sample_image(reflections, VK_SHADER_STAGE_FRAGMENT_BIT);

      input.result = builder.use_color_attachment(final_image, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(pipeline, 0);
      auto blk = cmd.allocate_ubo<Constants>();
      *blk.ptr = consts;

      auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

      gpu::write_set(set, 
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.material), sampler},
        gpu::TextureBinding {3, resources.get_view(input.diffuse), sampler},
        gpu::TextureBinding {4, resources.get_view(input.specular), sampler},
        gpu::TextureBinding {5, resources.get_view(input.reflections), sampler},
        gpu::UBOBinding {6, cmd.get_ubo_pool(), blk});
      
      const auto &image_info = resources.get_image(input.result)->get_extent();
      auto w = image_info.width;
      auto h = image_info.height;

      cmd.set_framebuffer(w, h, {resources.get_image_range(input.result)});
      cmd.bind_pipeline(pipeline);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_descriptors_graphics(0, {set}, {0});
      cmd.push_constants_graphics(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    });
}

void LightResolvePass::ui() {
  ImGui::Begin("LightResolvePass");
  ImGui::SliderFloat("Reflectiveness", &reflectiveness, 0.f, 1.f);
  ImGui::End();
}