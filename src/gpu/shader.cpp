#include "shader.hpp"

#include <iostream>
#include <fstream>
#include <cassert>

#include <sstream>
#include <initializer_list>

namespace gpu {
  

  DescriptorPool::DescriptorPool(uint32_t flips_count) 
  {
    VkDescriptorPoolSize sizes[] {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 512},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 128},
      {VK_DESCRIPTOR_TYPE_SAMPLER, 512}
    };

    VkDescriptorPoolCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .maxSets = 512,
      .poolSizeCount = sizeof(sizes)/sizeof(sizes[0]),
      .pPoolSizes = sizes
    };
    
    pools.resize(flips_count);
    
    for (uint32_t i = 0; i < flips_count; i++) {
      VKCHECK(vkCreateDescriptorPool(internal::app_vk_device(), &info, nullptr, &pools[i]));
    }
  }
  
  DescriptorPool::~DescriptorPool() {
    for (uint32_t i = 0; i < pools.size(); i++) {
      vkDestroyDescriptorPool(internal::app_vk_device(), pools[i], nullptr);
    }
  }

  void DescriptorPool::flip() {
    index = (index + 1) % pools.size();
    VKCHECK(vkResetDescriptorPool(internal::app_vk_device(), pools[index], 0));
  }
  
  void DescriptorPool::allocate_sets(uint32_t sets_count, const VkDescriptorSetLayout *set_layouts, VkDescriptorSet *sets, void *ext) {
    
    VkDescriptorSetAllocateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = ext,
      .descriptorPool = pools[index],
      .descriptorSetCount = sets_count,
      .pSetLayouts = set_layouts
    };

    VKCHECK(vkAllocateDescriptorSets(internal::app_vk_device(), &info, sets));
  }

  StaticDescriptorPool::StaticDescriptorPool() {
    VkDescriptorPoolSize sizes[] {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 512},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 128},
      {VK_DESCRIPTOR_TYPE_SAMPLER, 512}
    };

    VkDescriptorPoolCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 512,
      .poolSizeCount = sizeof(sizes)/sizeof(sizes[0]),
      .pPoolSizes = sizes
    };

    VKCHECK(vkCreateDescriptorPool(internal::app_vk_device(), &info, nullptr, &pool));
  }
  
  StaticDescriptorPool::~StaticDescriptorPool() {
    vkDestroyDescriptorPool(internal::app_vk_device(), pool, nullptr);
  }

  StaticDescriptorPool::StaticDescriptorPool(StaticDescriptorPool &&rhs)
    : pool {rhs.pool}
  {
    rhs.pool = nullptr;
  }

  void StaticDescriptorPool::allocate_sets(uint32_t sets_count, const VkDescriptorSetLayout *set_layouts, VkDescriptorSet *sets) {
    VkDescriptorSetAllocateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = pool,
      .descriptorSetCount = sets_count,
      .pSetLayouts = set_layouts
    };

    VKCHECK(vkAllocateDescriptorSets(internal::app_vk_device(), &info, sets));
  }
  
  VkDescriptorSet StaticDescriptorPool::allocate_set(VkDescriptorSetLayout layout, uint32_t variable_sizes_count, const uint32_t *variable_sizes) {
    VkDescriptorSetVariableDescriptorCountAllocateInfo ext {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorSetCount = variable_sizes_count,
      .pDescriptorCounts = variable_sizes
    };
    
    VkDescriptorSetAllocateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = variable_sizes_count? &ext : nullptr,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &layout
    };

    VkDescriptorSet set;
    VKCHECK(vkAllocateDescriptorSets(internal::app_vk_device(), &info, &set));
    return set;
  }

  void StaticDescriptorPool::free_sets(uint32_t sets_count, const VkDescriptorSet *sets) {
    VKCHECK(vkFreeDescriptorSets(internal::app_vk_device(), pool, sets_count, sets));
  }

  ManagedDescriptorSet::ManagedDescriptorSet(StaticDescriptorPool &desc_pool, VkDescriptorSetLayout layout, uint32_t variable_sizes_count, const uint32_t *variable_sizes) {
    reset(desc_pool, layout, variable_sizes_count, variable_sizes);
  }
  
  ManagedDescriptorSet::ManagedDescriptorSet(ManagedDescriptorSet &&rhs) 
    : pool {rhs.pool}, set {rhs.set}
  {
    rhs.pool = nullptr;
    rhs.set = nullptr;
  }
  
  ManagedDescriptorSet::~ManagedDescriptorSet() {
    internal_free();
  }

  void ManagedDescriptorSet::reset(StaticDescriptorPool &desc_pool, VkDescriptorSetLayout layout, uint32_t variable_sizes_count, const uint32_t *variable_sizes) {
    internal_free();
    pool = &desc_pool;
    set = pool->allocate_set(layout, variable_sizes_count, variable_sizes);
  }

  void ManagedDescriptorSet::internal_free() {
    if (pool && set) {
      pool->free_sets(1, &set);
    }
    pool = nullptr;
    set = nullptr;
  }

  ManagedDescriptorSet &ManagedDescriptorSet::operator=(ManagedDescriptorSet &&rhs) {
    pool = rhs.pool;
    set = rhs.set;
    rhs.pool = nullptr;
    rhs.set = nullptr;
    return *this;
  }

}