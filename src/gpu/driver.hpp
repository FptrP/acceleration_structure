#ifndef DRIVER_HPP_INCLUDED
#define DRIVER_HPP_INCLUDED

#include <lib/volk.h>
#include <lib/vk_mem_alloc.h>
#include <set>
#include <string>
#include <functional>

#include "common.hpp"

namespace gpu {

  struct Device;
  struct DebugMessenger;
  struct Swapchain;
  struct Swapchain;

  struct InstanceConfig {
    std::string app_name = "NOAPP";
    std::string engine_name = "NOENGINE";
    
    uint32_t app_version = VK_MAKE_VERSION(1, 0, 0);
    uint32_t engine_version = VK_MAKE_VERSION(1, 0, 0);
    uint32_t api_version = VK_API_VERSION_1_0;

    std::set<std::string> extensions;
    std::set<std::string> layers;
  };

  struct DeviceConfig {
    VkSurfaceKHR surface {nullptr};
    std::set<std::string> extensions;
    bool use_ray_query = false;
  };

  struct Instance {
    Instance(const InstanceConfig &cfg);
    Instance(Instance &&instance) : handle {instance.handle} { instance.handle = nullptr; }
    ~Instance();

    Device create_device(const DeviceConfig &cfg);
    DebugMessenger create_debug_messenger(PFN_vkDebugUtilsMessengerCallbackEXT callback);

    const Instance &operator=(Instance &&instance) { std::swap(handle, instance.handle); return *this; }
    
    const Instance &operator=(const Instance &instance) = delete;
    Instance(const Instance&) = delete;
  
    VkInstance api_instance() const { return handle; }
  private:
    VkInstance handle {nullptr};
  };
  
  struct DebugMessenger {
    DebugMessenger(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT callback);
    ~DebugMessenger();

    DebugMessenger(DebugMessenger &&msg) : base {msg.base}, handle {msg.handle} { msg.base = nullptr; msg.handle = nullptr; }
    
    const DebugMessenger &operator=(DebugMessenger &&msg) { 
      std::swap(handle, msg.handle); 
      std::swap(base, msg.base);
      return *this;
    }

    DebugMessenger(const DebugMessenger&) = delete;
    const DebugMessenger &operator=(const DebugMessenger &msg) = delete;
  private:
    VkInstance base {nullptr};
    VkDebugUtilsMessengerEXT handle {nullptr};
  };

  struct Device {
    Device(VkInstance instance, const DeviceConfig &cfg);
    Device(Device &&dev);
    ~Device();

    const Device &operator=(Device &&dev);

    Device(Device&) = delete;
    const Device &operator=(const Device &dev) = delete;
    
    VkDevice api_device() const { return logical_device; }
    VkQueue api_queue() const { return queue; }
    VkPhysicalDevice api_physical_device() const { return physical_device; }
    uint32_t get_queue_family() const { return queue_family_index; }
    VmaAllocator get_allocator() const { return allocator; }
    const VkPhysicalDeviceProperties get_properties() const { return properties; }

  private:
    VkPhysicalDevice physical_device {nullptr};
    VkPhysicalDeviceProperties properties;
    VkDevice logical_device {nullptr};
    VmaAllocator allocator {};

    uint32_t queue_family_index;
    VkQueue queue {nullptr};
  };

  struct Surface {
    Surface(VkInstance instance, VkSurfaceKHR surface) : base {instance}, handle {surface} {}
    ~Surface() { 
      if (base && handle) {
        vkDestroySurfaceKHR(base, handle, nullptr);
      }
    }

    Surface(Surface &&o) : base {o.base}, handle {o.handle} { o.handle = nullptr; }

    VkSurfaceKHR api_surface() const { return handle; }

    Surface &operator=(Surface &&o) {
      std::swap(base, o.base);
      std::swap(handle, o.handle);
      return *this;
    }

  private:
    VkInstance base {nullptr};
    VkSurfaceKHR handle {nullptr};

    Surface(const Surface&) = delete;
    const Surface &operator=(const Surface&) = delete;
  };

  using SurfaceCreateCB = std::function<VkSurfaceKHR(VkInstance)>;

  void create_context(const InstanceConfig &icfg, PFN_vkDebugUtilsMessengerCallbackEXT callback, DeviceConfig dcfg, SurfaceCreateCB &&surface_cb);
  void close_context();

  Instance &app_instance();
  Device &app_device();
  Surface &app_surface();

  struct QueueInfo {
    VkQueue queue;
    uint32_t family;
  };

  QueueInfo app_main_queue();

  namespace internal {
    VkDevice app_vk_device();
  }
}

#endif