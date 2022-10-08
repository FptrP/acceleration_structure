#include "descriptors.hpp"
#include "shader.hpp"

namespace gpu {

  static inline bool is_bufer_desc(VkDescriptorType type) {
    switch (type) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      return true;
    default:
      return false;
    }
  
    return false;
  }

  static inline bool is_image_desc(VkDescriptorType type) {
    switch (type) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      return true;
    default:
      return false;
    }
  
    return false;
  }

  static inline bool is_as_desc(VkDescriptorType type) {
    return type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  }

  static inline bool is_dynamic_buffer(VkDescriptorType type) {
    return type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
        || type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
  }

  void UpdateState::update_sets(VkDevice device) {
    vkUpdateDescriptorSets(device, writes_count, writes.data(), copies_count, copies.data());
  }

  void DescriptorSetState::clear() {
    modified_slots.set();
    valid_slots.reset();
    max_slots_used = 0;

    dyn_offset_count = 0;
    dyn_offsets = nullptr;

    for (auto &slot : slots) {
      slot.type = VK_DESCRIPTOR_TYPE_SAMPLER;
      slot.count = 0;
      slot.binding = 0;
      slot.as_state = nullptr;
      slot.buffer_state = nullptr;
      slot.image_state = nullptr;
      slot.dyn_offset = nullptr;
    }
  }
  
  /*void DescriptorSetState::init(const DescriptorSetResources &set, BinderAllocator &allocator) {
    if (set.bindings_count() > MAX_RES_SLOTS)
      throw std::runtime_error {"Too many resource bingings"};
    
    modified_slots.set();
    valid_slots.reset();
    max_slots_used = 0;

    dyn_offset_count = 0;
    dyn_offsets = nullptr;

    for (auto &elem : set) {
      valid_slots.set(elem.binding);
      max_slots_used = (elem.binding + 1 > max_slots_used)? (elem.binding + 1) : max_slots_used;

      auto &dst = slots[elem.binding];
      dst.binding = elem.binding;
      dst.type = elem.descriptorType;
      dst.count = elem.descriptorCount;
      
      dst.image_state = nullptr;
      dst.buffer_state = nullptr;
      dst.as_state = nullptr;
      dst.dyn_offset = nullptr;

      if (is_image_desc(dst.type)) {
        dst.image_state = allocator.alloc<VkDescriptorImageInfo>(dst.count);
        memset(dst.image_state, 0, sizeof(VkDescriptorImageInfo) * dst.count);
      }
      else if (is_bufer_desc(dst.type)) {
        dst.buffer_state = allocator.alloc<VkDescriptorBufferInfo>(dst.count);
        memset(dst.buffer_state, 0, sizeof(VkDescriptorBufferInfo) * dst.count);

        if (is_dynamic_buffer(dst.type)) {
          dyn_offset_count += dst.count;
        }
      }
      else if (is_as_desc(dst.type)) {
        dst.as_state = allocator.alloc<VkAccelerationStructureKHR>(dst.count);
        memset(dst.as_state, 0, sizeof(VkAccelerationStructureKHR) * dst.count);
      }
      else {
        throw std::runtime_error {"Unsupported descriptor type"};
      }
    }

    //allocate dynamic offsets
    if (dyn_offset_count) {
      dyn_offsets = allocator.alloc<uint32_t>(dyn_offset_count);
      memset(dyn_offsets, 0, sizeof(dyn_offsets[0]) * dyn_offset_count);

      auto ptr = dyn_offsets;
      for (uint32_t i = 0; i < max_slots_used; i++) {
        if (!valid_slots.test(i) || !is_dynamic_buffer(slots[i].type))
          continue;

        auto &slot = slots[i];
        slot.dyn_offset = ptr;
        ptr += slot.count;
      }
    }
  }*/
  
  void DescriptorSetState::update(UpdateState &state, VkDescriptorSet new_set, VkDescriptorSet old_set) {
    for (uint32_t i = 0; i < max_slots_used; i++) {
      if (!valid_slots.test(i))
        continue;
      
      const auto &slot = slots[i];
      
      if (modified_slots.test(i) || (old_set == nullptr))  
        write_slot(state, slot, new_set);
      else
        copy_slot(state, slot, new_set, old_set);

      modified_slots.reset(i);
    }
  }

  void DescriptorSetState::write_slot(UpdateState &state, const DescriptorSetSlot &slot, VkDescriptorSet new_set) {
    auto &write = state.push_write();
    
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext = nullptr;
    write.dstSet = new_set;
    write.dstBinding = slot.binding;
    write.descriptorType = slot.type;
    write.dstArrayElement = 0;
    write.descriptorCount = slot.count;
    write.pImageInfo = slot.image_state;
    write.pBufferInfo = slot.buffer_state;
    
    if (is_as_desc(slot.type)) {
      auto ptr = state.alloc<VkWriteDescriptorSetAccelerationStructureKHR>(1);
      ptr->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
      ptr->pNext = nullptr;
      ptr->accelerationStructureCount = slot.count;
      ptr->pAccelerationStructures = slot.as_state;
    
      write.pNext = ptr;
    }
  }
  
  void DescriptorSetState::copy_slot(UpdateState &state, const DescriptorSetSlot &slot, VkDescriptorSet new_set, VkDescriptorSet old_set) {
    auto &copy = state.push_copy();
    copy.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
    copy.pNext = nullptr;
    
    copy.srcSet = old_set;
    copy.srcBinding = slot.binding;
    copy.srcArrayElement = 0;
    
    copy.dstSet = new_set;
    copy.dstBinding = slot.binding;
    copy.dstArrayElement = 0;

    copy.descriptorCount = slot.count;
  }

  DescriptorBinder::DescriptorBinder() {
    for (auto &set : sets) {
      set.state.reset(new DescriptorSetState {});
    }
  }

  void DescriptorBinder::init_base(const BasePipeline &pipeline) {
    /*auto new_layout = pipeline.get_pipeline_layout();
    if (layout == new_layout)
      return;

    layout = new_layout;
    max_used_sets = 0;
    binder_allocator.reset(); //free al memory

    auto &resources = pipeline.get_resources();
    auto &pipeline_sets = resources.get_resources();
    
    for (auto &pipeline_set : pipeline_sets) {
      auto set_id = pipeline_set.get_set_id();
      
      if (set_id > MAX_DESC_SLOTS)
        throw std::runtime_error {"Too many descriptor sets"};

      max_used_sets = (set_id + 1 > max_used_sets)? (set_id + 1) : max_used_sets;
      valid_sets.set(set_id);

      auto &set_slot = sets[set_id];
      set_slot.vkset = nullptr;
      set_slot.update = true;
      set_slot.rebind = true;
      set_slot.layout = resources.get_desc_layout(set_id);
      set_slot.state->init(pipeline_set, binder_allocator);
    }*/

  }
  
  void DescriptorBinder::flush(DescriptorPool &pool, VkCommandBuffer cmd) {    
    for (uint32_t i = 0; i < max_used_sets; i++) {
      if (!valid_sets.test(i))
        continue;
      
      auto &set_slot = sets[i];
      if (!set_slot.update)
        continue;

      auto new_set = pool.allocate_set(set_slot.layout);
      set_slot.state->update(update_state, new_set, set_slot.vkset);
      set_slot.vkset = new_set;
    }

    update_state.update_sets(internal::app_vk_device());

    for (uint32_t i = 0; i < max_used_sets; i++) {
      if (!valid_sets.test(i))
        continue;
      
      auto &set_slot = sets[i];
      if (!set_slot.update || !set_slot.rebind)
        continue;
      
      vkCmdBindDescriptorSets(cmd, pipeline_type, layout, i, 1, &set_slot.vkset, set_slot.state->dyn_offset_count, set_slot.state->dyn_offsets);

      set_slot.update = false;
      set_slot.rebind = false;
    }

  }

  void DescriptorBinder::clear() {
    valid_sets.reset();
    max_used_sets = 0;
    layout = nullptr;
    pipeline_type = VK_PIPELINE_BIND_POINT_GRAPHICS;
    
    update_state.reset();
    binder_allocator.reset();

    for (auto &set : sets) {
      set.state->clear();
      
      set.layout = nullptr;
      set.vkset = nullptr;
      set.rebind = false;
      set.update = false;
    }

  }

}