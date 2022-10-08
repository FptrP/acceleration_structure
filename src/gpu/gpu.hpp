#ifndef GPU_HPP_INCLUDED
#define GPU_HPP_INCLUDED

#include "driver.hpp"
#include "swapchain.hpp"
#include "shader.hpp"
#include "cmd_buffers.hpp"
#include "sync_primitive.hpp"
#include "dynbuffer.hpp"
#include "samplers.hpp"
#include "descriptors.hpp"
#include "resources.hpp"
#include "managed_resources.hpp"

#include <functional>

namespace gpu {

  void init_all(const InstanceConfig &icfg, PFN_vkDebugUtilsMessengerCallbackEXT callback, DeviceConfig dcfg, VkExtent2D window_size, SurfaceCreateCB &&surface_cb);
  void close();

  Swapchain &app_swapchain();
  PipelinePool &app_pipelines();

  GraphicsPipeline create_graphics_pipeline();
  ComputePipeline create_compute_pipeline();
  ComputePipeline create_compute_pipeline(const char *name);

  VkSampler create_sampler(const VkSamplerCreateInfo &info);

  void reload_shaders();

  template <typename... Bindings> 
  void write_set(VkDescriptorSet set, const Bindings&... bindings) {
    internal::write_set(app_device().api_device(), set, bindings...);
  }
  
  void create_program(const std::string &name, std::initializer_list<std::string> shaders);
  void create_program(const std::string &name, std::vector<std::string> &&shaders);

  //std::vector<CmdContext> allocate_cmd_contexts(CmdBufferPool &pool, uint32_t count);
  
  //std::vector<Image> get_swapchain_images();
  std::vector<ImagePtr> get_swapchain_image_ptr();
  
  uint32_t get_swapchain_image_count();

  ManagedDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);
  ManagedDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout, const std::initializer_list<uint32_t> &variable_sizes);

  void collect_resources();
}

#endif