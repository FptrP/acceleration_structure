#include "gpu.hpp"
#include "framebuffers.hpp"

#include <optional>
#include <memory>

namespace gpu {
  
  static std::optional<Swapchain> g_swapchain;
  static std::unique_ptr<PipelinePool> g_pipeline_pool;
  static std::optional<SamplerPool> g_sampler_pool;
  static std::optional<StaticDescriptorPool> g_static_descriptors;

  void init_all(const InstanceConfig &icfg, PFN_vkDebugUtilsMessengerCallbackEXT callback, DeviceConfig dcfg, VkExtent2D window_size, SurfaceCreateCB &&surface_cb) {
    create_context(icfg, callback, dcfg, std::move(surface_cb));
    
    g_swapchain.emplace(Swapchain {
      window_size,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT});

    g_pipeline_pool.reset(new PipelinePool {});
    g_sampler_pool.emplace(SamplerPool {});
    g_static_descriptors.emplace(StaticDescriptorPool {});
  }

  void close() {
    vkDeviceWaitIdle(app_device().api_device());

    auto ptr = g_pipeline_pool.get();
    delete ptr;

    g_static_descriptors.reset();
    g_pipeline_pool.release();
    g_sampler_pool.reset();
    destroy_resources();
    
    g_swapchain.reset();
    close_context();
  }

  Swapchain &app_swapchain() {
    return g_swapchain.value();
  }
  
  PipelinePool &app_pipelines() {
    return *g_pipeline_pool;
  }

  GraphicsPipeline create_graphics_pipeline() {
    return {g_pipeline_pool.get()};
  }

  ComputePipeline create_compute_pipeline() {
    return {g_pipeline_pool.get()};
  }

  ComputePipeline create_compute_pipeline(const char *name) {
    gpu::ComputePipeline pipeline {g_pipeline_pool.get()};
    pipeline.set_program(name);
    return pipeline;
  }

  VkSampler create_sampler(const VkSamplerCreateInfo &info) {
    return g_sampler_pool->get_sampler(info);
  }
  
  void create_program(const std::string &name, std::initializer_list<std::string> shaders) {
    g_pipeline_pool->create_program(name, shaders);
  }
  
  void create_program(const std::string &name, std::vector<std::string> &&shaders) {
    g_pipeline_pool->create_program(name, std::move(shaders));
  }

  /*std::vector<CmdContext> allocate_cmd_contexts(CmdBufferPool &pool, uint32_t count) {
    auto &dev = app_device();
    auto api_buffers = pool.allocate(count);
    std::vector<CmdContext> cmd;
    cmd.reserve(count);
    for (auto elem : api_buffers) {
      cmd.emplace_back(elem, dev.get_properties().limits.minUniformBufferOffsetAlignment);
    }
    return cmd;
  }*/
  
  DescriptorPool new_descriptor_pool(uint32_t flips_count) {
    return DescriptorPool {flips_count};
  }
  
  CmdBufferPool new_command_pool() {
    return CmdBufferPool {};
  }
  
  Semaphore new_semaphore() {
    return Semaphore {};
  }
  
  Fence new_fence(bool signaled) {
    return Fence {signaled};
  }

  /*std::vector<Image> get_swapchain_images() {
    auto &ctx_dev = app_device();
    auto &ctx_swapchain = app_swapchain();
    auto images_count = get_swapchain_image_count();

    std::vector<VkImage> api_images;
    api_images.resize(images_count);
    VKCHECK(vkGetSwapchainImagesKHR(ctx_dev.api_device(), ctx_swapchain.api_swapchain(), &images_count, api_images.data()));

    std::vector<Image> images;
    images.reserve(images_count);
    for (auto handle : api_images) {
      images.emplace_back();
      images.back().create_reference(handle, ctx_swapchain.get_image_info());
    }
    return images;
  }*/

  std::vector<ImagePtr> get_swapchain_image_ptr() {
    auto &ctx_dev = app_device();
    auto &ctx_swapchain = app_swapchain();
    auto images_count = get_swapchain_image_count();

    std::vector<VkImage> api_images;
    api_images.resize(images_count);
    VKCHECK(vkGetSwapchainImagesKHR(ctx_dev.api_device(), ctx_swapchain.api_swapchain(), &images_count, api_images.data()));
    
    std::vector<ImagePtr> images;
    images.reserve(images_count);

    auto binfo = ctx_swapchain.get_image_info();
    VkImageCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = binfo.format,
      .extent = binfo.extent3D(),
      .mipLevels = binfo.mip_levels,
      .arrayLayers = binfo.array_layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    for (auto handle : api_images) {
      images.emplace_back(create_image_ref(handle, info));
    }
    return images;
  }
  
  uint32_t get_swapchain_image_count() {
    auto &swapchain = app_swapchain();
    return swapchain.get_images_count();
  }

  ManagedDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout) {
    return allocate_descriptor_set(layout, {});
  }
  
  ManagedDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout, const std::initializer_list<uint32_t> &variable_sizes) {
    return ManagedDescriptorSet {*g_static_descriptors, layout, variable_sizes.size(), variable_sizes.begin()};
  }

  void reload_shaders() {
    VKCHECK(vkDeviceWaitIdle(internal::app_vk_device()));
    g_pipeline_pool->reload_programs();
  }

  void collect_resources() {
    collect_image_buffer_resources();
  }
}