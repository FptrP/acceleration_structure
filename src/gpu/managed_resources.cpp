#include "managed_resources.hpp"

#include <cmath>

namespace gpu {

  //DriverResourceManager

  DriverResourceID DriverResourceManager::register_resource(DriverResource *res, bool acquire) {
    uint32_t slot = UINT32_MAX;
    std::scoped_lock lock {heap_lock};  
    if (free_list.size()){
      slot = free_list.back();
      free_list.pop_back();
    } else {
      slot = resources.size();
      resources.push_back({res, 0});
      return DriverResourceID {slot, 0};
    }

    resources[slot].first = res;
    resources[slot].second++;
    if (acquire)
      res->add_ref();
    return DriverResourceID {slot, resources[slot].second};
  }
    
  DriverResource *DriverResourceManager::acquire_resource(const DriverResourceID &id) {
    std::scoped_lock lock {heap_lock};
    auto &slot = resources.at(id.index);
    if (slot.second != id.gen) {
      throw std::runtime_error {"Bad resource generation"};
    }

    slot.first->add_ref();
    return slot.first;
  }
  
  void DriverResourceManager::release_resource(const DriverResourceID &id) {
    std::scoped_lock lock {heap_lock};
    auto &slot = resources.at(id.index);
    if (slot.second != id.gen) {
      throw std::runtime_error {"Bad resource generation"};
    }
    
    uint32_t references = slot.first->dec_ref();
    
    if (references <= 1u) {
      free_list.push_back(id.index);
      kill_list.push_back(slot.first);
      slot.first = nullptr;
    }
  }

  void DriverResourceManager::collect_garbage() {
    std::scoped_lock lock {heap_lock};
    for (auto ptr : kill_list) {
      delete ptr;
    }
    kill_list.clear();
  }
  
  void DriverResourceManager::clear_all() {
    std::scoped_lock lock {heap_lock};
    
    for (auto ptr : kill_list) {
      delete ptr;
    }
    
    for (auto &p : resources) {
      if (p.first)
        delete p.first;
    }

    kill_list.clear();
    resources.clear();
  }

  static DriverResourceManager g_res_manager;

  ResourcePtr::ResourcePtr(const DriverResourceID &res) : id {res} {
    ptr = g_res_manager.acquire_resource(id);
  }

  ResourcePtr::ResourcePtr(const ResourcePtr &rp) : id {rp.id} {
    if (id.valid())
      ptr = g_res_manager.acquire_resource(id);
  }

  ResourcePtr::ResourcePtr(ResourcePtr &&rp) : id {rp.id}, ptr {rp.ptr} {
    rp.id = INVALID_ID;
    rp.ptr = nullptr;
  } 

  ResourcePtr::~ResourcePtr() {
    if (id.valid()) {
      g_res_manager.release_resource(id);
    }
  }

  ResourcePtr &ResourcePtr::operator=(const ResourcePtr &rp) {
    if (id.valid())
      g_res_manager.release_resource(id);

    id = rp.id;
    ptr = rp.ptr;

    if (id.valid())
      ptr = g_res_manager.acquire_resource(id);

    return *this;
  }

  void ResourcePtr::release() {
    if (id.valid())
      g_res_manager.release_resource(id);
    id = INVALID_ID;
    ptr = nullptr;
  }
  
  void ResourcePtr::reset(DriverResourceID &new_id) {
    if (id.valid())
      g_res_manager.release_resource(id);
    
    id = new_id;
    ptr = g_res_manager.acquire_resource(new_id);
  }

  DriverBuffer::DriverBuffer(VmaMemoryUsage memory, uint64_t buffer_size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo buffer_info {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = buffer_size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr
    };

    VmaAllocationCreateInfo alloc_info {};
    alloc_info.usage = memory;
    alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    VmaAllocationInfo info {};
    
    auto base = app_device().get_allocator();

    VKCHECK(vmaCreateBuffer(base, &buffer_info, &alloc_info, &handle, &allocation, &info));
    
    VkMemoryPropertyFlags mem_flags = 0;
    vmaGetMemoryTypeProperties(base, info.memoryType, &mem_flags);
    coherent = mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; 
    mapped_ptr = info.pMappedData;
    size = buffer_size;
  }

  DriverBuffer::~DriverBuffer() {
    auto base = app_device().get_allocator();

    vmaDestroyBuffer(base, handle, allocation);
    base = nullptr;
    handle = nullptr;
    allocation = nullptr;
    size = 0;
    mapped_ptr = nullptr;
    coherent = false;
  }

  void DriverBuffer::invalidate_mapped_memory() {
    auto base = app_device().get_allocator();
    vmaInvalidateAllocation(base, allocation, 0, VK_WHOLE_SIZE);
  }

  void DriverBuffer::flush(uint64_t offset, uint64_t size) {
    if (coherent) return;
    auto base = app_device().get_allocator();
    vmaFlushAllocation(base, allocation, offset, size);
  }

  VkDeviceAddress DriverBuffer::device_address() const {
    VkBufferDeviceAddressInfo info {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .pNext = nullptr,
      .buffer = handle
    };

    return vkGetBufferDeviceAddress(app_device().api_device(), &info);
  }

  static VkImageAspectFlagBits get_default_aspect(VkFormat fmt) {
    switch (fmt) {
      case VK_FORMAT_S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT;
      case VK_FORMAT_D16_UNORM:
      case VK_FORMAT_D16_UNORM_S8_UINT:
      case VK_FORMAT_D24_UNORM_S8_UINT:
      case VK_FORMAT_D32_SFLOAT:
      case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
      default:
        return VK_IMAGE_ASPECT_COLOR_BIT;  
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
  }

  DriverImage::DriverImage(const VkImageCreateInfo &info) {
    auto allocator = app_device().get_allocator();
    desc = info;
    
    VmaAllocationCreateInfo alloc_info {};
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VKCHECK(vmaCreateImage(allocator, &info, &alloc_info, &handle, &allocation, nullptr));
  }
  
  DriverImage::DriverImage(VkImage vk_image, const VkImageCreateInfo &info) {
    handle = vk_image;
    desc = info;
    allocation = nullptr;
  }
  
  DriverImage::~DriverImage() {
    destroy_views();
    
    if (allocation)
      vmaDestroyImage(app_device().get_allocator(), handle, allocation);
  }

  VkImageAspectFlagBits DriverImage::get_default_aspect() const {
    return gpu::get_default_aspect(desc.format);
  }
  
  VkImageAspectFlags DriverImage::get_full_aspect() const { //todo fix
    VkImageAspectFlags aspect = get_default_aspect();
    if (aspect != VK_IMAGE_ASPECT_COLOR_BIT)
      aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return aspect;
  }

  VkImageView DriverImage::get_view(ImageViewRange range) {
    std::lock_guard lock {views_lock};

    auto device = app_device().api_device();

    if (!range.aspect) {
      range.aspect = get_default_aspect();
    }

    auto iter = views.find(range);
    if (iter != views.end()) {
      return iter->second;
    }

    VkImageViewCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .image = handle,
      .viewType = range.type,
      .format = desc.format,
      .components = {},
      .subresourceRange = {range.aspect, range.base_mip, range.mips_count, range.base_layer, range.layers_count}
    };

    VkImageView view {nullptr};
    VKCHECK(vkCreateImageView(device, &info, nullptr, &view));
    views.insert({range, view});
    return view;
  }

  void DriverImage::destroy_views() {
    auto vkdev = app_device().api_device();

    std::lock_guard lock {views_lock};
    for (auto [range, view] : views) {
      vkDestroyImageView(vkdev, view, nullptr);
    }
  }

  BufferPtr create_buffer(VmaMemoryUsage memory, uint64_t buffer_size, VkBufferUsageFlags usage) {
    auto *dbuf = new DriverBuffer {memory, buffer_size, usage}; 
    auto id = g_res_manager.register_resource(dbuf, false);
    return BufferPtr {id};
  }

  ImagePtr create_tex2d(VkFormat fmt, uint32_t w, uint32_t h, uint32_t mips, VkImageUsageFlags usage) {
    VkImageCreateInfo image_info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = VkExtent3D {w, h, 1u},
      .mipLevels = mips,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    auto *dimg = new DriverImage {image_info};
    auto id = g_res_manager.register_resource(dimg, false);
    return ImagePtr {id};
  }
  
  ImagePtr create_tex2d_mips(VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    uint32_t mips = std::floor(std::log2f(std::max(w, h))) + 1u;
    
    VkImageCreateInfo image_info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = VkExtent3D {w, h, 1u},
      .mipLevels = mips,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    auto *dimg = new DriverImage {image_info};
    auto id = g_res_manager.register_resource(dimg, false);
    return ImagePtr {id};
  }
  
  ImagePtr create_tex2d_array(VkFormat fmt, uint32_t w, uint32_t h, uint32_t mips, uint32_t layers, VkImageUsageFlags usage) {
    VkImageCreateInfo image_info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = VkExtent3D {w, h, 1u},
      .mipLevels = mips,
      .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    auto *dimg = new DriverImage {image_info};
    auto id = g_res_manager.register_resource(dimg, false);
    return ImagePtr {id};
  }

  ImagePtr create_cubemap(VkFormat fmt, uint32_t size, uint32_t mips, VkImageUsageFlags usage) {
    VkImageCreateInfo image_info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = VkExtent3D {size, size, 1u},
      .mipLevels = mips,
      .arrayLayers = 6,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    auto *dimg = new DriverImage {image_info};
    auto id = g_res_manager.register_resource(dimg, false);
    return ImagePtr {id};
  }

  ImagePtr create_image_ref(VkImage vkimg, const VkImageCreateInfo &info) {
    auto *dimg = new DriverImage {vkimg, info};
    auto id = g_res_manager.register_resource(dimg, false);
    return ImagePtr {id};
  }

  ImagePtr create_driver_image(const VkImageCreateInfo &info) {
    auto *dimg = new DriverImage {info};
    auto id = g_res_manager.register_resource(dimg, false);
    return ImagePtr {id};
  }

  void collect_image_buffer_resources() {
    g_res_manager.collect_garbage();
  }

  void destroy_resources() {
    g_res_manager.clear_all();
  }

  DriverResource *acquire_resource(DriverResourceID id) {
    return g_res_manager.acquire_resource(id);
  }

  void release_resource(const DriverResourceID &id) {
    g_res_manager.release_resource(id);
  }

  ImagePtr acquire_image(DriverResourceID id) {
    return ImagePtr {id};
  }
  
  BufferPtr acquire_buffer(DriverResourceID id) {
    return BufferPtr {id};
  }

}