#ifndef GPU_MANAGED_RESOURCES_HPP_INCLUDED
#define GPU_MANAGED_RESOURCES_HPP_INCLUDED

#include <atomic>
#include <mutex>

#include "resource_info.hpp"
#include "driver.hpp"

namespace gpu {
  struct DriverResourceManager;

  struct DriverResourceID {
    constexpr DriverResourceID() {}
    constexpr DriverResourceID(const DriverResourceID &id) : index {id.index}, gen {id.gen} {}

    constexpr bool operator==(const DriverResourceID &id) const {
      return index == id.index && gen == id.gen;
    }

    constexpr bool operator!=(const DriverResourceID &id) const {
      return index != id.index || gen != id.gen;
    }

    DriverResourceID &operator=(const DriverResourceID &id) {
      index = id.index;
      gen = id.gen;
      return *this;
    }

    constexpr bool invalid() const { return index == UINT32_MAX || gen == UINT32_MAX; }
    constexpr bool valid() const { return index < UINT32_MAX && gen < UINT32_MAX; }
  private:
    DriverResourceID(uint32_t i, uint32_t g) : index {i}, gen {g} {}
    
    uint32_t index {UINT32_MAX};
    uint32_t gen {UINT32_MAX};

    friend DriverResourceManager;
    friend std::hash<DriverResourceID>;
  };

  constexpr DriverResourceID INVALID_ID {};

  struct DriverResource { 
    virtual ~DriverResource() {}

    uint32_t add_ref() { return references.fetch_add(1u); }
    uint32_t dec_ref() { return references.fetch_sub(1u); }
    uint32_t ref_count() const { return references.load(std::memory_order_seq_cst); }

  private:  
    std::atomic<uint32_t> references {0u};
  };

  struct DriverResourceManager {
    DriverResourceID register_resource(DriverResource *res, bool acquire);
    
    DriverResource *acquire_resource(const DriverResourceID &id);
    void release_resource(const DriverResourceID &id);

    void collect_garbage();
    void clear_all();
    
  private:
    std::mutex heap_lock;
    std::vector<uint32_t> free_list;
    std::vector<std::pair<DriverResource*, uint32_t>> resources;
    std::vector<DriverResource*> kill_list;
  };

  struct ResourcePtr {
    ResourcePtr() {}
    ResourcePtr(const DriverResourceID &res);
    ResourcePtr(const ResourcePtr &rp);
    ResourcePtr(ResourcePtr &&rp);

    ~ResourcePtr();
    
    operator bool() const { return id.valid(); }

    bool operator==(const ResourcePtr &rp) const { return rp.id == id && ptr == rp.ptr; }

    ResourcePtr &operator=(const ResourcePtr &rp);
    
    ResourcePtr &operator=(ResourcePtr &&rp) {
      std::swap(id, rp.id);
      std::swap(ptr, rp.ptr);
      return *this;
    }

    void release();
    void reset(DriverResourceID &id);

    DriverResourceID get_id() const { return id; }

  protected:
    DriverResourceID id;
    DriverResource *ptr {nullptr};
  };

  struct DriverBuffer : DriverResource {
    DriverBuffer(VmaMemoryUsage memory, uint64_t buffer_size, VkBufferUsageFlags usage);
    ~DriverBuffer();
    
    void flush(uint64_t offset = 0, uint64_t size = VK_WHOLE_SIZE);
    void invalidate_mapped_memory();
    
    VkBuffer api_buffer() const { return handle; }
    uint64_t get_size() const { return size; }
    bool is_coherent() const { return coherent; }
    
    void *get_mapped_ptr() const { return mapped_ptr; }

    VkDeviceAddress device_address() const;

    DriverBuffer(DriverBuffer&) = delete;
    const DriverBuffer &operator=(const DriverBuffer&) = delete;
  
  private:
    VkBuffer handle {nullptr};
    VmaAllocation allocation {nullptr};
    uint64_t size {0};
    bool coherent = false;
    void *mapped_ptr = nullptr;
  };

  struct DriverImage : DriverResource {
    DriverImage(const VkImageCreateInfo &info);
    DriverImage(VkImage vk_image, const VkImageCreateInfo &info);
    ~DriverImage();

    VkImage api_image() const { return handle; }
    VkExtent3D get_extent() const { return desc.extent; }
    VkFormat get_fmt() const { return desc.format; }
    uint32_t get_mip_levels() const { return desc.mipLevels; }
    uint32_t get_array_layers() const { return desc.arrayLayers; }
    const VkImageCreateInfo &get_info() const { return desc; }
    
    VkImageAspectFlagBits get_default_aspect() const;
    VkImageAspectFlags get_full_aspect() const;

    VkImageView get_view(ImageViewRange range);
    void destroy_views();

    DriverImage(const DriverImage &) = delete;
    DriverImage &operator=(const DriverImage &) = delete;

  private:
    VkImage handle {nullptr};
    VmaAllocation allocation {nullptr};
    VkImageCreateInfo desc;

    std::mutex views_lock;
    std::unordered_map<ImageViewRange, VkImageView> views;
  };

  struct BufferPtr : ResourcePtr {
    DriverBuffer* operator->() { return static_cast<DriverBuffer*>(ptr); }
    const DriverBuffer* operator->() const { return static_cast<const DriverBuffer*>(ptr); }
  };

  struct ImagePtr : ResourcePtr {
    DriverImage *operator->() { return static_cast<DriverImage*>(ptr); }
    const DriverImage *operator->() const { return static_cast<const DriverImage*>(ptr); }
  };

  void collect_image_buffer_resources();
  void destroy_resources();

  BufferPtr create_buffer(VmaMemoryUsage memory, uint64_t buffer_size, VkBufferUsageFlags usage);
  
  ImagePtr create_tex2d(VkFormat fmt, uint32_t w, uint32_t h, uint32_t mips, VkImageUsageFlags usage);
  ImagePtr create_tex2d_mips(VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage);
  ImagePtr create_tex2d_array(VkFormat fmt, uint32_t w, uint32_t h, uint32_t mips, uint32_t layers, VkImageUsageFlags usage);
  ImagePtr create_cubemap(VkFormat fmt, uint32_t size, uint32_t mips, VkImageUsageFlags usage);
  ImagePtr create_image_ref(VkImage vkimg, const VkImageCreateInfo &info);
  ImagePtr create_driver_image(const VkImageCreateInfo &info);

  DriverResource *acquire_resource(DriverResourceID id);
  void release_resource(const DriverResourceID &id);

  ImagePtr acquire_image(DriverResourceID id);
  BufferPtr acquire_buffer(DriverResourceID id);
}

namespace std {
  template <>
  struct hash<gpu::DriverResourceID> {
    size_t operator()(const gpu::DriverResourceID &key) const {
      size_t h = uint64_t(key.index) | (uint64_t(key.gen) << 31);
      return h;  
    }
  };
}

#endif