#ifndef GPU_SHADER_PROGRAM_HPP_INCLUDED
#define GPU_SHADER_PROGRAM_HPP_INCLUDED

#include "driver.hpp"

#include <bitset>
#include <unordered_map>

#include <lib/spirv-reflect/spirv_reflect.h>

namespace gpu {
  constexpr uint32_t MAX_BINDINGS = 16u;
  constexpr uint32_t MAX_DESCRIPTORS = 8u;

  inline bool operator==(const VkDescriptorSetLayoutBinding &a, const VkDescriptorSetLayoutBinding &b) {
    return a.binding == b.binding &&
           a.descriptorCount == b.descriptorCount &&
           a.descriptorType == b.descriptorType &&
           a.stageFlags == b.stageFlags &&
           a.pImmutableSamplers == b.pImmutableSamplers;
  }
  
  inline bool operator!=(const VkDescriptorSetLayoutBinding &a, const VkDescriptorSetLayoutBinding &b) {
    return !(a == b);
  }

  struct DescriptorSetLayoutHash;

  struct DescriptorSetLayoutInfo {
    void parse_resources(VkShaderStageFlagBits stage, const SpvReflectDescriptorSet *set);

    bool operator==(const DescriptorSetLayoutInfo &info) const {
      if (used_bindings != info.used_bindings)
        return false;
      if (valid_bindings != info.valid_bindings)
        return false;

      for (uint32_t i = 0; i < used_bindings; i++) {
        if (!valid_bindings.test(i))
          continue;

        if (bindings[i] != info.bindings[i])
          return false;
        if (flags[i] != info.flags[i])
          return false;
      }
      return true;
    }
    
    VkDescriptorSetLayout create_api_layout(VkDevice device) const;

    uint32_t get_used_bindings() const { return used_bindings; };
    bool has_binding(uint32_t index) const { return valid_bindings.test(index); }
    bool has_bindless_resources() const { return bindless_bindings; }

    const VkDescriptorSetLayoutBinding &get_binding(uint32_t index) const { return bindings.at(index); }
    VkDescriptorBindingFlags get_flags(uint32_t index) const { return flags.at(index); }

    
  private:
    std::array<VkDescriptorSetLayoutBinding, MAX_BINDINGS> bindings;
    std::array<VkDescriptorBindingFlags, MAX_BINDINGS> flags;
    uint32_t used_bindings = 0;
    std::bitset<MAX_BINDINGS> valid_bindings;

    bool bindless_bindings = false;
  
    friend DescriptorSetLayoutHash;
  };

  struct DescriptorSetLayoutHash {
    std::size_t operator()(const DescriptorSetLayoutInfo &res) const {
      std::size_t h = 0;
      for (uint32_t i = 0; i < res.used_bindings; i++) {
        if (!res.valid_bindings.test(i))
          continue;
        
        hash_combine(h, res.bindings[i].binding);
        hash_combine(h, res.bindings[i].descriptorType);
        hash_combine(h, res.bindings[i].descriptorCount);
        hash_combine(h, res.bindings[i].stageFlags);
        hash_combine(h, res.flags[i]);
      }
      return h;
    }
  };

  using DescriptorLayoutId = uint32_t;

  struct DescriptorSetLayoutCache {
    DescriptorSetLayoutCache() {}
    ~DescriptorSetLayoutCache() { clear(); }
    
    DescriptorLayoutId register_layout(const DescriptorSetLayoutInfo &info);
    void clear();

    const DescriptorSetLayoutInfo &get_layout_info(DescriptorLayoutId id) const {
      return desc_info.at(id);
    }

    VkDescriptorSetLayout get_layout(DescriptorLayoutId id) const {
      return vk_layouts.at(id);
    }

    DescriptorSetLayoutCache(const DescriptorSetLayoutCache&) = delete;
    DescriptorSetLayoutCache &operator=(const DescriptorSetLayoutCache&) = delete; 
  private:
    std::unordered_map<DescriptorSetLayoutInfo, DescriptorLayoutId, DescriptorSetLayoutHash> map;
    std::vector<DescriptorSetLayoutInfo> desc_info;
    std::vector<VkDescriptorSetLayout> vk_layouts;
  };

  struct ShaderModule {
    ShaderModule(const std::string_view &path);
    ShaderModule(ShaderModule &&mod);
    ~ShaderModule();

    void reload();

    const SpvReflectShaderModule &get_resources() const { return spv_module; }
    VkShaderModule get_module() const { return api_module; }
    VkShaderStageFlagBits get_stage() const { return static_cast<VkShaderStageFlagBits>(spv_module.shader_stage); }
    std::string_view get_name() const { return spv_module.entry_point_name; }

    ShaderModule &operator=(ShaderModule &&mod);

    ShaderModule(const ShaderModule &mod) = delete;
    ShaderModule &operator=(const ShaderModule &mod) = delete;
  private:
    std::string path {};
    VkShaderModule api_module {nullptr};
    SpvReflectShaderModule spv_module {};
  };

  using ShaderProgramId = uint32_t;

  struct ShaderProgramManager;

  struct ShaderProgramManager {
    ShaderProgramManager() {}
    ~ShaderProgramManager() { clear(); }
    
    ShaderProgramId create_program(const std::string &name, const std::vector<std::string> &shaders); 
    
    ShaderProgramId get_program(const std::string &name) const;

    void reload();
    void clear();

    VkPipelineLayout get_program_layout(ShaderProgramId id) const;
    const std::bitset<MAX_DESCRIPTORS> &get_used_descriptors(ShaderProgramId id) const;
    const DescriptorSetLayoutInfo &get_program_descriptor_info(ShaderProgramId id, uint32_t set) const;
    VkDescriptorSetLayout get_program_descriptor_layout(ShaderProgramId id, uint32_t set) const;
    std::vector<VkPipelineShaderStageCreateInfo> get_stage_info(ShaderProgramId id) const; 

    ShaderProgramManager(const ShaderProgramManager&) = delete;
    ShaderProgramManager &operator=(const ShaderProgramManager&) = delete;
  private:
    DescriptorSetLayoutCache cached_descriptors;

    std::unordered_map<std::string, uint32_t> module_names;
    std::vector<ShaderModule> modules;

    struct ShaderProgInternal;
    std::unordered_map<std::string, ShaderProgramId> prog_names;
    std::vector<ShaderProgInternal> programs;

    uint32_t load_module(const std::string &name);
    ShaderModule &get_module(uint32_t id) { return modules.at(id); }
    const ShaderModule &get_module(uint32_t id) const { return modules.at(id); }

    void reset_program(ShaderProgInternal &prog);
    void destroy_program(ShaderProgInternal &prog);
    void validate_program_shaders(const std::vector<uint32_t> mod_ids);

    struct ShaderProgInternal {
      std::vector<uint32_t> modules;
      
      std::bitset<MAX_DESCRIPTORS> valid_sets;
      std::array<DescriptorLayoutId, MAX_DESCRIPTORS> sets;
      
      VkPushConstantRange constants {0u, 0u, 0u};
      VkPipelineLayout layout {nullptr};
    };
  };

  struct ShaderProgram {
    ShaderProgram(ShaderProgramManager &manager, ShaderProgramId pid) : mgr {&manager}, id {pid} {}

    VkPipelineLayout get_layout() const { return mgr->get_program_layout(id); }
    

  private:
    ShaderProgramManager *mgr {nullptr};
    ShaderProgramId id;
  };

}

#endif