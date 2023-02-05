#include "driver.hpp"

#include <vector>

#define VMA_IMPLEMENTATION
#include <lib/vk_mem_alloc.h>
#include <optional>

namespace gpu {

  Instance::Instance(const InstanceConfig &cfg) {
    VKCHECK(volkInitialize());

    VkApplicationInfo app_info {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = cfg.app_name.c_str(),
      .applicationVersion = cfg.app_version,
      .pEngineName = cfg.engine_name.c_str(),
      .engineVersion = cfg.engine_version,
      .apiVersion = cfg.api_version
    };
  
    std::vector<const char*> layers;
    std::vector<const char*> extensions;

    layers.reserve(cfg.layers.size());
    for (const auto &s : cfg.layers) {
      layers.push_back(s.c_str());
    }

    extensions.reserve(cfg.extensions.size());
    for (auto &s : cfg.extensions) {
      extensions.push_back(s.c_str());
    }

    VkInstanceCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = (uint32_t)layers.size(),
      .ppEnabledLayerNames = layers.data(),
      .enabledExtensionCount = (uint32_t)extensions.size(),
      .ppEnabledExtensionNames = extensions.data()
    };

    VKCHECK(vkCreateInstance(&info, nullptr, &handle));
    
    volkLoadInstance(handle);
  }

  Instance::~Instance() {
    if (handle) {
      vkDestroyInstance(handle, nullptr);
    }
  }

  Device Instance::create_device(const DeviceConfig &cfg) {
    return {handle, cfg};
  }
  
  DebugMessenger Instance::create_debug_messenger(PFN_vkDebugUtilsMessengerCallbackEXT callback) {
    return {handle, callback};
  }

  DebugMessenger::DebugMessenger(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT callback)
    : base {instance}
  {
    VkDebugUtilsMessengerCreateInfoEXT info {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .pNext = nullptr,
      .flags = 0,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = callback,
      .pUserData = nullptr
    };
      
    VKCHECK(vkCreateDebugUtilsMessengerEXT(instance, &info, nullptr, &handle));
  }
  
  DebugMessenger::~DebugMessenger() {
    if (base && handle) {
      vkDestroyDebugUtilsMessengerEXT(base, handle, nullptr);
    }
  }


  struct DeviceQueryInfo {
    bool complete = false;
    uint32_t queue_family_index = 0;
    VkPhysicalDeviceProperties properties;
  };

  static DeviceQueryInfo pick_physical_device(VkPhysicalDevice device, const DeviceConfig &cfg) {
    VkPhysicalDeviceProperties pproperties;
    vkGetPhysicalDeviceProperties(device, &pproperties);

    if (pproperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      return {false, 0};
    }

    uint32_t count = 0;
    std::vector<VkQueueFamilyProperties> queues;

    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    queues.resize(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queues.data());

    bool queue_found = false;
    uint32_t queue_family = 0;

    for (uint32_t i = 0; i < queues.size(); i++) {
      auto flags = queues[i].queueFlags;
      uint32_t required = VK_QUEUE_COMPUTE_BIT|VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_TRANSFER_BIT;
      auto msk = flags & required;
        
      if (msk != required) {
        continue;
      }

      if (cfg.surface) {
        VkBool32 supported = VK_FALSE;
        VKCHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, cfg.surface, &supported));

        if (!supported) {
          continue;
        }
      }

      queue_found = true;
      queue_family = i;
    }

    return {queue_found, queue_family, pproperties};
  }

  Device::Device(VkInstance instance, const DeviceConfig &cfg) {
    uint32_t count = 0;
    std::vector<VkPhysicalDevice> pdevices;

    VKCHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    pdevices.resize(count);
    VKCHECK(vkEnumeratePhysicalDevices(instance, &count, pdevices.data()));

    DeviceQueryInfo query {};
    for (auto pdev : pdevices) {
      query = pick_physical_device(pdev, cfg);
      if (query.complete) {
        physical_device = pdev;
        properties = query.properties;
        break;
      }
    }

    queue_family_index = query.queue_family_index;    

    float priority = 1.f;
    
    VkDeviceQueueCreateInfo queues[] {
      {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &priority 
      }
    };

    auto ext_set = cfg.extensions;
    
    if (cfg.use_ray_query) {
      //ext_set.insert(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
      //ext_set.insert(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
      ext_set.insert("VK_KHR_deferred_host_operations");
      ext_set.insert("VK_KHR_acceleration_structure");
      ext_set.insert("VK_KHR_ray_query");
    }

    std::vector<const char*> extensions;
    extensions.reserve(ext_set.size());
    for (auto &s : ext_set) {
      extensions.push_back(s.c_str());
    }

    VkPhysicalDeviceFeatures features {};
    features.fragmentStoresAndAtomics = VK_TRUE;
    features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    features.tessellationShader = VK_TRUE;
    features.geometryShader = VK_TRUE;
    
    VkPhysicalDeviceDescriptorIndexingFeatures bindless_features {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
      .pNext = nullptr
    };

    VkPhysicalDeviceBufferDeviceAddressFeatures device_adders {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      nullptr,
      VK_TRUE,
      VK_FALSE,
      VK_FALSE
    };

    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
    };
    acceleration_structure.accelerationStructure = VK_TRUE;
    acceleration_structure.accelerationStructureCaptureReplay = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR ray_query_featrues {};
    ray_query_featrues.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    ray_query_featrues.rayQuery = VK_TRUE;
    ray_query_featrues.pNext = nullptr;
    acceleration_structure.pNext = &ray_query_featrues;
    device_adders.pNext = &acceleration_structure;

    bindless_features.runtimeDescriptorArray = VK_TRUE;
    bindless_features.descriptorBindingPartiallyBound = VK_TRUE;
    bindless_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    bindless_features.pNext = cfg.use_ray_query? &device_adders : nullptr;
    
    VkDeviceCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &bindless_features,
      .flags = 0,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = queues,
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = nullptr,
      .enabledExtensionCount = (uint32_t)extensions.size(),
      .ppEnabledExtensionNames = extensions.data(),
      .pEnabledFeatures = &features
    };

    VKCHECK(vkCreateDevice(physical_device, &info, nullptr, &logical_device));
    vkGetDeviceQueue(logical_device, queue_family_index, 0, &queue);
  
    VmaVulkanFunctions vk_func {
      vkGetInstanceProcAddr,
      vkGetDeviceProcAddr,
      vkGetPhysicalDeviceProperties,
      vkGetPhysicalDeviceMemoryProperties,
      vkAllocateMemory,
      vkFreeMemory,
      vkMapMemory,
      vkUnmapMemory,
      vkFlushMappedMemoryRanges,
      vkInvalidateMappedMemoryRanges,
      vkBindBufferMemory,
      vkBindImageMemory,
      vkGetBufferMemoryRequirements,
      vkGetImageMemoryRequirements,
      vkCreateBuffer,
      vkDestroyBuffer,
      vkCreateImage,
      vkDestroyImage,
      vkCmdCopyBuffer,
    };

    VmaAllocatorCreateInfo alloc_info {};
    alloc_info.device = logical_device;
    alloc_info.instance = instance;
    alloc_info.physicalDevice = physical_device;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_2;
    alloc_info.pVulkanFunctions = &vk_func;
    if (cfg.use_ray_query) {
      alloc_info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    
    VKCHECK(vmaCreateAllocator(&alloc_info, &allocator));
  }
  
  Device::Device(Device &&dev)
    : physical_device {dev.physical_device}, properties {dev.properties}, logical_device {dev.logical_device},
      allocator{dev.allocator}, queue_family_index {dev.queue_family_index},
      queue {dev.queue}
  {
    dev.logical_device = nullptr;
    dev.allocator = nullptr;
  }

  Device::~Device() {
    if (allocator) {
      vmaDestroyAllocator(allocator);
    }
    if (logical_device) {
      vkDestroyDevice(logical_device, nullptr);
    }
  }

  const Device &Device::operator=(Device &&dev) {
    std::swap(physical_device, dev.physical_device);
    std::swap(properties, dev.properties);
    std::swap(logical_device, dev.logical_device);
    std::swap(allocator, dev.allocator);
    std::swap(queue_family_index, dev.queue_family_index);
    std::swap(queue, dev.queue);
    return *this;
  }

  static std::optional<Instance> g_instance;
  static std::optional<DebugMessenger> g_messenger;
  static std::optional<Surface> g_surface;
  static std::optional<Device> g_device;

  void create_context(const InstanceConfig &icfg, PFN_vkDebugUtilsMessengerCallbackEXT callback, DeviceConfig dcfg, SurfaceCreateCB &&surface_cb) {
    g_instance.emplace(Instance {icfg});

    if (callback) {
      g_messenger.emplace(g_instance->create_debug_messenger(callback));
    }
    
    if (surface_cb) {
      auto api_surface = surface_cb(g_instance->api_instance());
      g_surface.emplace(Surface {g_instance->api_instance(), api_surface});
      dcfg.surface = g_surface->api_surface();
    }
    
    g_device.emplace(Device {g_instance->api_instance(), dcfg});
  }
  
  void close_context() {
    g_device.reset();
    g_surface.reset();
    g_messenger.reset();
    g_instance.reset();
  }

  Instance &app_instance() {
    return g_instance.value();
  }

  Device &app_device() {
    return g_device.value();
  }

  Surface &app_surface() {
    return g_surface.value();
  }

  VkDevice internal::app_vk_device() {
    return g_device.value().api_device();
  }

  QueueInfo app_main_queue() {
    auto &dev = app_device();
    return QueueInfo {dev.api_queue(), dev.get_queue_family()};
  }
  
}