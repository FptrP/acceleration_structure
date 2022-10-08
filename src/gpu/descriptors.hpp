#ifndef DESCRIPTORS_HPP_INCLUDED
#define DESCRIPTORS_HPP_INCLUDED

#include "resources.hpp"
#include "dynbuffer.hpp"
#include "pipelines.hpp"
#include "managed_resources.hpp"

#include <bitset>

namespace gpu {
  struct BaseBinding;

  namespace internal {
    inline void write_set_base(VkDevice api_device, VkDescriptorSet set, VkWriteDescriptorSet *ptr, const BaseBinding &b0);
  }

  struct BaseBinding {
    BaseBinding(uint32_t base_binding, uint32_t dst_array_elem, uint32_t desc_count, VkDescriptorType type) {
      desc_write.dstBinding = base_binding;
      desc_write.dstArrayElement = dst_array_elem;
      desc_write.descriptorCount = desc_count; 
      desc_write.descriptorType = type;
    }
  protected:
    VkWriteDescriptorSet desc_write { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };

    friend inline void internal::write_set_base(VkDevice api_device, VkDescriptorSet set, VkWriteDescriptorSet *ptr, const BaseBinding &b0);
  };

  struct UBOBinding : BaseBinding {
    template<typename T>
    UBOBinding(uint32_t binding, const UniformBufferPool &pool, const UboBlock<T> &)
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC}
    {
      info.buffer = pool.api_buffer();
      info.offset = 0;
      info.range = sizeof(T);
      desc_write.pBufferInfo = &info;
    }

    UBOBinding(uint32_t binding, const UniformBufferPool &pool, uint64_t size)
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC}
    {
      info.buffer = pool.api_buffer();
      info.offset = 0;
      info.range = size;
      desc_write.pBufferInfo = &info;
    }

    UBOBinding(uint32_t binding, const gpu::BufferPtr &buf) 
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC}
    {
      info.buffer = buf->api_buffer();
      info.offset = 0;
      info.range = buf->get_size();

      desc_write.pBufferInfo = &info;
    }

  private:
    VkDescriptorBufferInfo info {};
  };

  struct SSBOBinding : BaseBinding {

    SSBOBinding(uint32_t binding, const gpu::BufferPtr &buf) 
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    {
      info.buffer = buf->api_buffer();
      info.offset = 0;
      info.range = buf->get_size();

      desc_write.pBufferInfo = &info;
    }

    SSBOBinding(uint32_t binding, VkBuffer buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE) 
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
    {
      info.buffer = buffer;
      info.offset = offset;
      info.range = range;
      desc_write.pBufferInfo = &info;
    }


  private:
    VkDescriptorBufferInfo info {};
  };

  struct TextureBinding : BaseBinding {
    TextureBinding(uint32_t binding, VkImageView view, VkSampler sampler)
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}
    {
      info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      info.sampler = sampler;
      info.imageView = view;
      desc_write.pImageInfo = &info;
    }

  private:
    VkDescriptorImageInfo info {};
  };

  struct StorageTextureBinding : BaseBinding {
    StorageTextureBinding(uint32_t binding, VkImageView view)
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}
    {
      info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      info.sampler = nullptr;
      info.imageView = view;
      desc_write.pImageInfo = &info;
    }

  private:
    VkDescriptorImageInfo info {};
  };

  struct ArrayOfImagesBinding : BaseBinding {
    ArrayOfImagesBinding(uint32_t binding, const std::vector<VkImageView> &src)
      : BaseBinding {binding, 0, (uint32_t)src.size(), VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE}
    {
      view_info.reserve(src.size());
      for (auto elem : src) {
        VkDescriptorImageInfo info {
          .sampler = nullptr,
          .imageView = elem,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        };
        view_info.push_back(info);
      }

      desc_write.pImageInfo = view_info.data();
    }

    ArrayOfImagesBinding(uint32_t binding, const std::vector<std::pair<VkImageView, VkSampler>> &src)
      : BaseBinding {binding, 0, (uint32_t)src.size(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}
    {
      view_info.reserve(src.size());
      for (auto elem : src) {
        VkDescriptorImageInfo info {
          .sampler = elem.second,
          .imageView = elem.first,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        };
        view_info.push_back(info);
      }

      desc_write.pImageInfo = view_info.data();
    }

  private:
    std::vector<VkDescriptorImageInfo> view_info;
  };

  struct SamplerBinding : BaseBinding {
    SamplerBinding(uint32_t binding, VkSampler sampler)
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER}
    {
      info.sampler = sampler;
      desc_write.pImageInfo = &info;
    }

  private:
    VkDescriptorImageInfo info {};
  };

  struct AccelerationStructBinding : BaseBinding {
    AccelerationStructBinding(uint32_t binding, VkAccelerationStructureKHR tlas)
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR}
    {
      tlas_copy = tlas;
      info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
      info.pNext = nullptr;
      info.accelerationStructureCount = 1;
      info.pAccelerationStructures = &tlas_copy;

      desc_write.pNext = &info;
    }
  private:
    VkAccelerationStructureKHR tlas_copy;
    VkWriteDescriptorSetAccelerationStructureKHR info {}; 
  };

  namespace internal {
    inline void write_set_base(VkDevice api_device, VkDescriptorSet set, VkWriteDescriptorSet *ptr, const BaseBinding &b0) {
      *ptr = b0.desc_write;
      ptr->dstSet = set;
    }
  
    template <typename... Bindings>
    void write_set_base(VkDevice api_device, VkDescriptorSet set, VkWriteDescriptorSet *ptr, const BaseBinding &b0, const Bindings&... rest)
    {
      write_set_base(api_device, set, ptr, b0);
      write_set_base(api_device, set, ptr + 1, rest...);
    }

    template <typename... Bindings> 
    void write_set(VkDevice api_device, VkDescriptorSet set, const Bindings&... bindings) {
      constexpr auto count = sizeof...(bindings);

      VkWriteDescriptorSet writes[count];
      write_set_base(api_device, set, writes, bindings...);

      vkUpdateDescriptorSets(api_device, count, writes, 0, nullptr);
    }
  }

  inline bool operator!=(const VkDescriptorImageInfo &a, const VkDescriptorImageInfo &b) {
    return (a.sampler != b.sampler) || (a.imageView != b.imageView) || (a.imageLayout != b.imageLayout);
  } 

  inline bool operator!=(const VkDescriptorBufferInfo &a, const VkDescriptorBufferInfo &b) {
    return (a.buffer != b.buffer) || (a.offset != b.offset) || (a.range != b.range);
  }

  template <size_t Bytes>
  struct StaticAllocator {
    void reset() {
      offset = 0; 
    }

    template<typename T>
    T *alloc(uint32_t elems) {
      uint32_t bytes = elems * sizeof(T);
      if (offset + bytes > Bytes)
        throw std::runtime_error {"OOM"};
      
      auto ptr = reinterpret_cast<T*>(storage.data() + offset);
      offset += bytes;
      return ptr;
    }

  private:
    std::array<uint8_t, Bytes> storage {};
    uint32_t offset = 0;
  };

  constexpr uint32_t MAX_RES_SLOTS = 16;
  constexpr uint32_t MAX_DESC_SLOTS = 8;
  constexpr uint32_t MAX_RES_HEAP = 10 * (1 << 10); //10Kb
  constexpr uint32_t MAX_UPDATE_RES = MAX_RES_SLOTS * MAX_DESC_SLOTS; //20Kb;

  using BinderAllocator = StaticAllocator<MAX_RES_HEAP>;
  //using UpdateAllocator = StaticAllocator<MAX_UPDATE_HEAP>;

  struct DescriptorSetSlot {
    uint32_t binding {0u};
    uint32_t count {0u};
    VkDescriptorType type;

    VkDescriptorImageInfo *image_state;
    VkDescriptorBufferInfo *buffer_state;
    VkAccelerationStructureKHR *as_state;

    uint32_t *dyn_offset = nullptr; //for dynamic uniform buffers
  };

  struct UpdateState {
    void reset() {
      writes_count = 0;
      copies_count = 0;
      mem_alloc.reset();
    }

    VkWriteDescriptorSet &push_write() {
      if (writes_count >= MAX_UPDATE_RES)
        throw std::runtime_error {"OOM"};
      return writes[writes_count];
    }

    VkCopyDescriptorSet &push_copy() {
      if (copies_count >= MAX_UPDATE_RES)
        throw std::runtime_error {"OOM"};
      return copies[copies_count++];
    }

    template<typename T>
    T *alloc(uint32_t count) { return mem_alloc.alloc<T>(count); }

    void update_sets(VkDevice device);

  private:
    StaticAllocator<MAX_RES_HEAP> mem_alloc;
    std::array<VkWriteDescriptorSet, MAX_UPDATE_RES> writes;
    std::array<VkCopyDescriptorSet,  MAX_UPDATE_RES> copies;
    uint32_t writes_count = 0;
    uint32_t copies_count = 0;
  };

  struct DescriptorBinder;

  struct DescriptorSetState {
    bool set(uint32_t binding, uint32_t array_index, VkSampler sampler, VkImageView view, VkImageLayout layout) {
      if (binding >= max_slots_used || !valid_slots.test(binding))
        throw std::runtime_error {"Invalid slot access"};
      
      auto &slot = slots[binding];
      if (slot.count <= array_index)
        throw std::runtime_error {"Array index out of bounds"};

      auto &dst = slot.image_state[array_index];
      VkDescriptorImageInfo info {sampler, view, layout};
      bool update = info != dst;

      dst = info;
      if (update)
        modified_slots.set(binding, true);
      return update;
    }
    
    bool set(uint32_t binding, uint32_t array_index, VkBuffer buffer, size_t offset, size_t range) {
      if (binding >= max_slots_used || !valid_slots.test(binding))
        throw std::runtime_error {"Invalid slot access"};
      
      auto &slot = slots[binding];
      if (slot.count <= array_index)
        throw std::runtime_error {"Array index out of bounds"};

      auto &dst = slot.buffer_state[array_index];
      VkDescriptorBufferInfo info {buffer, offset, range};
      bool update = info != dst;

      dst = info;
      if (update)
        modified_slots.set(binding, true);
      return update;
    }

    bool set_dynbuffer_offset(uint32_t binding, uint32_t array_index, uint32_t offset) {
      if (binding >= max_slots_used || !valid_slots.test(binding))
        throw std::runtime_error {"Invalid slot access"};
      
      auto &slot = slots[binding];
      if (slot.dyn_offset == nullptr)
        throw std::runtime_error {"Resource does not support dynamic offsets"};
      if (array_index >= slot.count)
        throw std::runtime_error {"Array index out of bounds"};

      bool rebind = slot.dyn_offset[array_index] != offset;
      slot.dyn_offset[array_index] = offset;
      return rebind;
    }

    //void init(const DescriptorSetResources &set, BinderAllocator &allocator);
    void update(UpdateState &state, VkDescriptorSet new_set, VkDescriptorSet old_set);
    void clear();
    
  private:
    std::array<DescriptorSetSlot, MAX_RES_SLOTS> slots;
    std::bitset<MAX_RES_SLOTS> modified_slots;
    std::bitset<MAX_RES_SLOTS> valid_slots;
    uint32_t max_slots_used = 0;

    uint32_t *dyn_offsets = nullptr;
    uint32_t  dyn_offset_count = 0;

    void write_slot(UpdateState &state, const DescriptorSetSlot &slot, VkDescriptorSet new_set);
    void copy_slot(UpdateState &state, const DescriptorSetSlot &slot, VkDescriptorSet new_set, VkDescriptorSet old_set);

    friend DescriptorBinder;
  };
  
  struct DescriptorPool;

  struct DescriptorBinder {
    DescriptorBinder();
    ~DescriptorBinder() {}

    void set(uint32_t set, uint32_t binding, uint32_t array_index, VkSampler sampler, VkImageView view, VkImageLayout layout) {
      if (set >= max_used_sets || !valid_sets.test(set))
        throw std::runtime_error {"Set out of bounds"};
      sets[set].update |= sets[set].state->set(binding, array_index, sampler, view, layout);
    }
    
    void set(uint32_t set, uint32_t binding, uint32_t array_index, VkBuffer buffer, size_t offset, size_t range) {
      if (set >= max_used_sets || !valid_sets.test(set))
        throw std::runtime_error {"Set out of bounds"};
      sets[set].update |= sets[set].state->set(binding, array_index, buffer, offset, range);
    }

    void set_dynbuffer_offset(uint32_t set, uint32_t binding, uint32_t array_index, uint32_t offset) {
      if (set >= max_used_sets || !valid_sets.test(set))
        throw std::runtime_error {"Set out of bounds"};
      sets[set].rebind |= sets[set].state->set_dynbuffer_offset(binding, array_index, offset);
    }

    void init(const ComputePipeline &pipeline) {
      init_base(pipeline);
      pipeline_type = VK_PIPELINE_BIND_POINT_COMPUTE;
    }

    void init(const GraphicsPipeline &pipeline) {
      init_base(pipeline);
      pipeline_type = VK_PIPELINE_BIND_POINT_GRAPHICS;
    }

    void flush(DescriptorPool &pool, VkCommandBuffer cmd);
    void clear();

    DescriptorBinder(const DescriptorBinder &) = delete;
    DescriptorBinder &operator=(const DescriptorBinder &) = delete;
  private:

    struct DescData {
      std::unique_ptr<DescriptorSetState> state;
      VkDescriptorSetLayout layout = nullptr;
      VkDescriptorSet vkset = nullptr; 
      bool update = false;
      bool rebind = false;
    };  

    std::array<DescData, MAX_DESC_SLOTS> sets;
    std::bitset<MAX_DESC_SLOTS> valid_sets;
    uint32_t max_used_sets = 0;

    VkPipelineLayout layout = nullptr;
    VkPipelineBindPoint pipeline_type = VK_PIPELINE_BIND_POINT_GRAPHICS;

    UpdateState update_state;
    BinderAllocator binder_allocator;
  
    void init_base(const BasePipeline &pipeline);
  };
  
}

#endif