#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <filesystem>
#include <lib/json.hpp>
#include <ctime>

using json = nlohmann::json; 
namespace fs = std::filesystem;

#include "gpu/gpu.hpp"
#include "scene/scene.hpp"
#include "scene/scene_as.hpp"
#include "rendergraph/rendergraph.hpp"

#include "backbuffer_subpass2.hpp"
#include "util_passes.hpp"
#include "scene_renderer.hpp"
#include "defered_shading.hpp"
#include "gpu_transfer.hpp"
#include "downsample_pass.hpp"
#include "ssr.hpp"
#include "gtao.hpp"
#include "trace_samples.hpp"
#include "image_readback.hpp"
#include "advanced_ssr.hpp"
#include "taa.hpp"
#include "depth_as.hpp"
#include "rtfx.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <lib/stb_image_write.h>

#define ENABLE_VALIDATION 1
#define USE_RAY_QUERY 1

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(
  VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  VkDebugUtilsMessageTypeFlagsEXT messageType,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  void* pUserData)  
{
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
}

struct AppInit {
  AppInit(uint32_t width, uint32_t height, bool enable_validation) {
    SDL_Init(SDL_INIT_EVERYTHING);
    window = SDL_CreateWindow("T", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_VULKAN);

    uint32_t count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr);
    std::vector<const char*> ext; 
    ext.resize(count);
    SDL_Vulkan_GetInstanceExtensions(window, &count, ext.data());

    gpu::InstanceConfig instance_info {};
    instance_info.api_version = VK_API_VERSION_1_2;
    if (enable_validation) {
      instance_info.layers = {"VK_LAYER_KHRONOS_validation"};
    }
    instance_info.extensions.insert(ext.begin(), ext.end());
    instance_info.extensions.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    gpu::DeviceConfig device_info {};
#if USE_RAY_QUERY
    device_info.use_ray_query = true;
#endif
    device_info.extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    gpu::init_all(instance_info, debug_cb, device_info, {width, height}, [&](VkInstance instance){
      VkSurfaceKHR surface;
      SDL_Vulkan_CreateSurface(window, instance, &surface);
      return surface;
    });
  }

  ~AppInit() {
    gpu::close();
    SDL_DestroyWindow(window);
    SDL_Quit();
  }

  SDL_Window *window;
};

static glm::vec4 next_taa_offset(uint32_t w, uint32_t h) {
  static int index = 0;
  
  constexpr glm::vec2 offsets[] {
    {0.25f, 0.25f},
    {0.75f, 0.75f},
    {0.75f, 0.25f},
    {0.25f, 0.75f}
  };

  glm::vec2 inv_resolution {1.0/w, 1.0/h};
  glm::vec2 offset = 2.0f * offsets[index] - 1.0f;
  index = (index + 1) % 4; 

  return glm::vec4{offset * inv_resolution, 0.f, 0.f};
}

static void get_depth_cb(ReadBackData &&image) {
  const uint32_t *word_ptr = reinterpret_cast<const uint32_t*>(image.bytes.get());
  
  const char *save_path = "captures/gbuffer_depth.csv";
  std::ofstream file {save_path, std::ios::trunc};
  file << "y, ";

  for (uint32_t x = 0; x < image.width; x++) {
    file << x;
    if (x != image.width - 1) {
      file << ",";
    }
  }
  
  file << "\n";

  for (uint32_t y = 0; y < image.height; y++) {
    file << y << ",";
    file << std::hex;

    for (uint32_t x = 0; x < image.width; x++) {
      auto depth = 0xffffff & word_ptr[y * image.width + x];
      file << "0x" << depth;
      if (x != image.width - 1) {
        file << ",";
      }
    }
    
    file << std::dec;
    file << "\n";
  }
  file.close();
}

static void get_depth_png_cb(ReadBackData &&image) {
  uint32_t *word_ptr = reinterpret_cast<uint32_t*>(image.bytes.get());
  for (uint32_t y = 0; y < image.height; y++) {
    for (uint32_t x = 0; x < image.width; x++) {
      word_ptr[y * image.width + x] &= 0xffffff;
    }
  }
  int res = stbi_write_png("captures/gbuffer_depth.png", int(image.width), int(image.height), 4, image.bytes.get(), 0);
  std::cout << "STBI: " << res << "\n";
}

static void get_rgba_cb(ReadBackData &&image) {
  for (uint32_t y = 0; y < image.height; y++) {
    for (uint32_t x = 0; x < image.width; x++) {
      uint32_t offset = y * image.width + x;
      image.bytes[4 * offset + 3] = 255;
    }
  }

  int t = clock();
  std::string name = std::string{"captures/gbuffer_color_"} + std::to_string(t) + ".png";
  //int res = stbi_write_png("captures/gbuffer_color.png", int(image.width), int(image.height), 4, image.bytes.get(), 0);
  int res = stbi_write_png(name.c_str(), int(image.width), int(image.height), 4, image.bytes.get(), 0);
  std::cout << "STBI: " << res << "\n";
}

static void load_shaders(const fs::path &config_path) {
  const fs::path shader_dir = config_path.parent_path();

  std::ifstream conf_file {config_path};
  auto config = json::parse(conf_file);
  
  const std::unordered_map<std::string, VkShaderStageFlagBits> stages_map {
    {"vertex", VK_SHADER_STAGE_VERTEX_BIT},
    {"fragment", VK_SHADER_STAGE_FRAGMENT_BIT},
    {"compute", VK_SHADER_STAGE_COMPUTE_BIT}
  };

  for (const auto &elem : config.items()) {
    const auto prog_name = elem.key();
    const auto &prog = elem.value();

    std::vector<std::string> shader_names;
    shader_names.reserve(prog.size());

    for (const auto &shader : prog.items()) {
      const auto &key = shader.key();
      const auto &val = shader.value();
      auto stage = stages_map.find(key); 
      if (stage == stages_map.end()) {
        throw std::runtime_error {"Incorrect stage"};
      }

      auto file_path = shader_dir / val.get<std::string>();
      if (!file_path.has_extension()) {
        file_path += ".spv";
      }

      shader_names.push_back(file_path.string());
    }
    std::cout << "Loading " << prog_name << " program\n";
    gpu::create_program(prog_name, std::move(shader_names));
  }
}

//const uint32_t WIDTH = 1280;
//const uint32_t HEIGHT = 720;
const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;
//const uint32_t WIDTH = 2560;
//const uint32_t HEIGHT = 1440;

rendergraph::ImageResourceId create_readbackimage(rendergraph::RenderGraph &graph) {
  gpu::ImageInfo image_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, WIDTH, HEIGHT};
  return graph.create_image(VK_IMAGE_TYPE_2D, image_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

int main(int argc, char **argv) {
  bool enable_validation = true;
  
  std::vector<std::string> params;
  params.reserve(argc - 1);
  for (int i = 1; i < argc; i++) {
    params.push_back(argv[i]);
  }

  if (params.size() > 0 && params[0] == "--disable-validation") {
    std::cout << "validation disabled\n";
    enable_validation = false;
  }
  
  AppInit app_init {WIDTH, HEIGHT, enable_validation};
  load_shaders("src/shaders/config.json");

  auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
  bool use_jitter = true;
  
  rendergraph::RenderGraph render_graph {gpu::app_device(), gpu::app_swapchain()};
  gpu_transfer::init(render_graph);
  ReadBackSystem readback_system;

  gpu::TransferCmdPool transfer_pool {};
  auto scene = scene::load_tinygltf_scene(transfer_pool,  "assets/gltf/Sponza/glTF/Sponza.gltf", USE_RAY_QUERY);
  //auto scene = scene::load_tinygltf_scene(transfer_pool,  "/home/void/workspace/tools/glTF-Sample-Models/room/room_gltf/roomgltf.gltf", USE_RAY_QUERY);
  //auto scene = scene::load_tinygltf_scene(transfer_pool,  "assets/gltf/st_dragon/stanford-dragon.gltf", USE_RAY_QUERY);
  //auto scene = scene::load_tinygltf_scene(transfer_pool,  "assets/gltf/sibernik_gltf/untitled.gltf", USE_RAY_QUERY);

#if USE_RAY_QUERY
  bool use_rt_ao = false;
  scene::SceneAccelerationStructure acceleration_struct;
  acceleration_struct.build(transfer_pool, scene);
#endif

  SamplesMarker::init(render_graph, WIDTH, HEIGHT);
  
  Gbuffer gbuffer {render_graph, WIDTH, HEIGHT};
  DownsamplePass downsample_pass {};
  GTAO gtao {render_graph, WIDTH, HEIGHT, USE_RAY_QUERY, 1};
  AdvancedSSR ssr {render_graph, WIDTH, HEIGHT};
  TAA taa_pass {render_graph, WIDTH, HEIGHT};
  RTReflections rt_reflections {render_graph, WIDTH/2, HEIGHT/2};

  ssr.preintegrate_pdf(render_graph);
  ssr.preintegrate_brdf(render_graph);

  SceneRenderer scene_renderer {scene};
  scene_renderer.init_pipeline(render_graph, gbuffer);
  DeferedShadingPass shading_pass {render_graph, app_init.window};  
  
  //DepthAs screen_space_as;
  //DepthAsBuilder screen_space_as_builder;

  //screen_space_as.create(transfer_pool, WIDTH/2, HEIGHT/2);
  //screen_space_as_builder.init(WIDTH/2, HEIGHT/2);

  imgui_create_fonts(transfer_pool);

  auto readback_image = create_readbackimage(render_graph);
  auto shadows_tex = render_graph.create_image(
    VK_IMAGE_TYPE_2D, 
    gpu::ImageInfo{
      VK_FORMAT_D24_UNORM_S8_UINT, 
      VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT, 
      1024, 1024, 1, 1, 4
    }, 
    VK_IMAGE_TILING_OPTIMAL, 
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  auto color_out_tex = render_graph.create_image(VK_IMAGE_TYPE_2D,
    gpu::ImageInfo {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, WIDTH, HEIGHT},
    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  scene::Camera camera({0.f, 1.f, -1.f});  
  glm::mat4 projection = glm::perspective(glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f);
  glm::mat4 shadow_mvp = glm::perspective(glm::radians(90.f), 1.f, 0.05f, 80.f) * glm::lookAt(glm::vec3{-1.85867, 5.81832, -0.247114}, glm::vec3{0, 2, 1}, glm::vec3{0, -1, 0});

  DrawTAAParams draw_params {};
  draw_params.prev_mvp = projection * camera.get_view_mat();
  draw_params.camera = camera.get_view_mat();

  bool quit = false;
  auto ticks = SDL_GetTicks();
  
  render_graph.submit();

  clear_depth(render_graph, gbuffer.prev_depth);

  glm::mat4 prev_mvp = projection * camera.get_view_mat();
  ReadBackID image_read_back = INVALID_READBACK;
  bool reload_request = false;
  while (!quit) {
    imgui_new_frame();
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      imgui_handle_event(event);
      
      if (event.type == SDL_QUIT) {
        quit = true;
      } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_r) {
        reload_request = true;
      } 

      camera.process_event(event);
    }

    auto ticks_now = SDL_GetTicks();
    float dt = (ticks_now - ticks)/1000.f;
    ticks = ticks_now;

    camera.move(dt);
    draw_params.prev_mvp = draw_params.mvp;
    draw_params.mvp = projection * camera.get_view_mat();
    draw_params.prev_camera = draw_params.camera;
    draw_params.camera = camera.get_view_mat();
    draw_params.fovy_aspect_znear_zfar = glm::vec4{glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f};
    draw_params.jitter = use_jitter? next_taa_offset(gbuffer.w, gbuffer.h) : glm::vec4{0.f, 0.f, 0.f, 0.f};

    scene_renderer.update_scene();
    shading_pass.update_params(camera.get_view_mat(), shadow_mvp, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f);
    
    gpu_transfer::process_requests(render_graph);

    SamplesMarker::clear(render_graph);

    //scene_renderer.draw_taa(render_graph, gbuffer, draw_params);
    scene_renderer.rasterize_triange_id(render_graph, gbuffer, draw_params);
    scene_renderer.reconstruct_gbuffer(render_graph, gbuffer, draw_params);

    downsample_pass.run(render_graph, gbuffer.normal, gbuffer.velocity_vectors, gbuffer.depth, gbuffer.downsampled_normals, gbuffer.downsampled_velocity_vectors);

    //screen_space_as_builder.run(render_graph, screen_space_as, gbuffer.depth, 1, draw_params);

    ImGui::Begin("Read texture");
    bool depth = ImGui::Button("Depth") && (image_read_back == INVALID_READBACK);
    bool color = ImGui::Button("Color") && (image_read_back == INVALID_READBACK);
    bool output_readback = ImGui::Button("Final frame") && (image_read_back == INVALID_READBACK);
    if (depth) {
      image_read_back = readback_system.read_image(render_graph, gbuffer.depth, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0);
    } else if (color) {
      image_read_back = readback_system.read_image(render_graph, gbuffer.albedo);
    }
    ImGui::Checkbox("Enable jitter", &use_jitter);
#if USE_RAY_QUERY
    ImGui::Checkbox("Enable RT AO", &use_rt_ao);
#endif
    ImGui::End();

    ssr.render_ui();
    gtao.draw_ui();
    shading_pass.draw_ui();

    auto normal_mat = glm::transpose(glm::inverse(camera.get_view_mat()));
    auto camera_to_world = glm::inverse(camera.get_view_mat());
    GTAOParams gtao_params {normal_mat, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f};
    GTAORTParams gtao_rt_params {camera_to_world, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f};
    AdvancedSSRParams assr_params {normal_mat, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f};    
    
    ssr.run(render_graph, assr_params, draw_params, gbuffer, gtao.raw);
    
    
#if USE_RAY_QUERY
    if (use_rt_ao) {
      gtao.add_main_rt_pass(render_graph, gtao_rt_params, acceleration_struct.tlas, gbuffer.depth, gbuffer.normal);
    } else 
#endif
    {
      gtao.add_main_pass(render_graph, gtao_params, gbuffer.depth, gbuffer.normal, gbuffer.material, ssr.get_preintegrated_pdf());
    }

    gtao.add_filter_pass(render_graph, gtao_params, gbuffer.depth);
    gtao.add_accumulate_pass(render_graph, draw_params, gbuffer);

    shading_pass.draw(render_graph, gbuffer, shadows_tex, gtao.accumulated_ao, ssr.get_preintegrated_brdf(), ssr.get_blurred(), color_out_tex);
    taa_pass.run(render_graph, gbuffer, color_out_tex, draw_params);
    
    if (output_readback && image_read_back == INVALID_READBACK) {
      blit_image(render_graph, taa_pass.get_output(), readback_image);
      image_read_back = readback_system.read_image(render_graph, readback_image);
    }
    
    //rt_reflections.run(render_graph, gbuffer, screen_space_as, assr_params);

    add_backbuffer_subpass(render_graph, taa_pass.get_output(), sampler, DrawTex::ShowAll);
    //add_backbuffer_subpass(render_graph, gtao.accumulated_ao, sampler, DrawTex::ShowR);
    //add_backbuffer_subpass(render_graph, ssr.get_blurred(), sampler, DrawTex::ShowAll);
    add_present_subpass(render_graph);
    render_graph.submit();
    readback_system.after_submit(render_graph);

    if (image_read_back != INVALID_READBACK && readback_system.is_data_available(image_read_back)) {
      auto data = readback_system.get_data(image_read_back);
      if (data.texel_fmt == VK_FORMAT_D24_UNORM_S8_UINT) {
        get_depth_cb(std::move(data));
      } else {
        get_rgba_cb(std::move(data));
      }
      
      image_read_back = INVALID_READBACK;
    }

    render_graph.remap(gbuffer.depth, gbuffer.prev_depth);
    render_graph.remap(gtao.output, gtao.prev_frame);
    taa_pass.remap_targets(render_graph);
    ssr.remap_images(render_graph);
    gtao.remap(render_graph);
    prev_mvp = projection * camera.get_view_mat();

    if (reload_request) {
      gpu::reload_shaders();
      reload_request = false;
    }

    gpu::collect_resources();
  }
  
  vkDeviceWaitIdle(gpu::app_device().api_device());
  gpu_transfer::close();
  imgui_close();
  return 0;
}