#include "shader_program.hpp"

#include <stdexcept>
#include <fstream>
#include <sstream>

namespace gpu {
  
  void DescriptorSetLayoutInfo::parse_resources(VkShaderStageFlagBits stage, const SpvReflectDescriptorSet *set) {
    for (uint32_t i = 0u; i < set->binding_count; i++) {
      const auto &spv_binding = *set->bindings[i];
      if (spv_binding.binding >= MAX_BINDINGS)
        throw std::runtime_error {"Too many bindings"};


      uint32_t spv_binding_count = 1;
      for (uint32_t i = 0; i < spv_binding.array.dims_count; i++) {
        spv_binding_count *= spv_binding.array.dims[i];
      }

      auto spv_desc_type = static_cast<VkDescriptorType>(spv_binding.descriptor_type);
      if (spv_desc_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        spv_desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;

      bool spv_bindless = spv_binding_count == 0;
      if (spv_bindless) {
        spv_binding_count = 1024;
        bindless_bindings = true;
      }

      if (valid_bindings.test(spv_binding.binding)) {
        auto &api_binding = bindings[spv_binding.binding];

        if (api_binding.descriptorType != spv_desc_type)  {
          throw std::runtime_error {"Incompatible desc. type"};
        }

        if (api_binding.descriptorCount != spv_binding_count) {
          throw std::runtime_error {"Bindings count mismatch"};
        }

        api_binding.stageFlags |= stage;
        continue;
      }

      VkDescriptorSetLayoutBinding api_binding {};
      api_binding.binding = spv_binding.binding;
      api_binding.descriptorType = spv_desc_type;
      api_binding.stageFlags = stage;
      api_binding.descriptorCount = spv_binding_count;
      
      valid_bindings.set(spv_binding.binding);
      bindings[spv_binding.binding] = api_binding;
      flags[spv_binding.binding] = spv_bindless? (VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT|VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT) : 0;

      used_bindings = ((spv_binding.binding + 1) > used_bindings)? (spv_binding.binding + 1) : used_bindings; 
    }
  }

  VkDescriptorSetLayout DescriptorSetLayoutInfo::create_api_layout(VkDevice device) const {
    std::array<VkDescriptorSetLayoutBinding, MAX_BINDINGS> info_bindings;
    std::array<VkDescriptorBindingFlags, MAX_BINDINGS> info_flags;
    uint32_t elems_count = 0;

    for (uint32_t binding = 0; binding < used_bindings; binding++) {
      if (!valid_bindings.test(binding))
        continue;

      info_bindings[elems_count] = bindings[binding];
      info_flags[elems_count] = flags[binding];
      elems_count++;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .pNext = nullptr,
      .bindingCount = elems_count,
      .pBindingFlags = info_flags.data()
    };

    VkDescriptorSetLayoutCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &flags_info,
      .flags = 0,
      .bindingCount = elems_count,
      .pBindings = info_bindings.data()
    };

    VkDescriptorSetLayout layout {nullptr};
    VKCHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout));
    return layout;
  }

  DescriptorLayoutId DescriptorSetLayoutCache::register_layout(const DescriptorSetLayoutInfo &info) {
    auto it = map.find(info);
    if (it != map.end())
      return it->second;
    
    DescriptorLayoutId id = desc_info.size();
    map.insert({info, id});
    desc_info.push_back(info);
    vk_layouts.push_back(info.create_api_layout(internal::app_vk_device()));
    return id;
  }
  
  void DescriptorSetLayoutCache::clear() {
    for (auto layout : vk_layouts) {
      vkDestroyDescriptorSetLayout(internal::app_vk_device(), layout, nullptr);
    }

    map.clear();
    desc_info.clear();
    vk_layouts.clear();
  }

  static std::vector<char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      std::stringstream ss;
      ss << "Failed to open file " << filename;
      throw std::runtime_error {ss.str()};
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
  }

  ShaderModule::ShaderModule(const std::string_view &prog_path)
    : path {prog_path}
  {
    reload();
  }
  
  ShaderModule::ShaderModule(ShaderModule &&mod)
    : path {std::move(mod.path)}, api_module {mod.api_module}, spv_module {mod.spv_module}
  {
    mod.api_module = nullptr;
  }

  ShaderModule::~ShaderModule() {
    if (!api_module)
      return;

    vkDestroyShaderModule(internal::app_vk_device(), api_module, nullptr);
    spvReflectDestroyShaderModule(&spv_module);
  }

  void ShaderModule::reload() {
    if (api_module) {
      vkDestroyShaderModule(internal::app_vk_device(), api_module, nullptr);
      spvReflectDestroyShaderModule(&spv_module);
    }

    auto code = read_file(path);

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VKCHECK(vkCreateShaderModule(internal::app_vk_device(), &create_info, nullptr, &api_module));

    if (spvReflectCreateShaderModule(code.size(), code.data(), &spv_module) != SPV_REFLECT_RESULT_SUCCESS) {
      throw std::runtime_error {"Shader parsing error"};
    }
  }

  ShaderModule &ShaderModule::operator=(ShaderModule &&mod) {
    std::swap(path, mod.path);
    std::swap(api_module, mod.api_module);
    std::swap(spv_module, mod.spv_module);
    return *this;
  }


  uint32_t ShaderProgramManager::load_module(const std::string &name) {
    auto it = module_names.find(name);
    if (it != module_names.end()) {
      return it->second;
    }

    uint32_t index = module_names.size();
    modules.emplace_back(name);
    module_names.insert({name, index});
    return index;
  }

  ShaderProgramId ShaderProgramManager::get_program(const std::string &name) const {
    auto it = prog_names.find(name);
    if (it == prog_names.end()) {
      throw std::runtime_error {"Program not found"};
    }
    return it->second;
  }

  void ShaderProgramManager::validate_program_shaders(const std::vector<uint32_t> mod_ids) {
    auto supported_shaders = 
      VK_SHADER_STAGE_VERTEX_BIT|
      VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT|
      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT|
      VK_SHADER_STAGE_GEOMETRY_BIT|
      VK_SHADER_STAGE_FRAGMENT_BIT|
      VK_SHADER_STAGE_COMPUTE_BIT;
     bool is_compute_pipeline = false;
    uint32_t usage_mask = 0;

    for (const auto &id : mod_ids) {
      auto &mod = modules[id];
      if ((mod.get_stage() & supported_shaders) == 0) {
        throw std::runtime_error {"Error, unsupported shader"};
      }

      if (mod.get_stage() & usage_mask) {
        throw std::runtime_error {"Error, multiple usage of same shader stage"};
      }

      is_compute_pipeline |= mod.get_stage() == VK_SHADER_STAGE_COMPUTE_BIT;
      usage_mask |= mod.get_stage();
    }

    if (is_compute_pipeline && mod_ids.size() != 1) {
      throw std::runtime_error {"Error, usage of compute shader with other stages"};
    }
  }

  ShaderProgramId ShaderProgramManager::create_program(const std::string &name, const std::vector<std::string> &shaders) {
    if (prog_names.find(name) != prog_names.end())
      throw std::runtime_error {"Program already created"};
    
    ShaderProgInternal prog;
    prog.valid_sets.reset();
    prog.modules.reserve(shaders.size());
    prog.layout = nullptr;

    for (auto &shader_path : shaders) {
      prog.modules.emplace_back(load_module(shader_path));
    }
    
    std::sort(prog.modules.begin(), prog.modules.end(),
      [&](uint32_t mod_a, uint32_t mod_b){
        return modules[mod_a].get_stage() < modules[mod_b].get_stage();
      });

    validate_program_shaders(prog.modules);

    ShaderProgramId id = programs.size();
    prog_names[name] = id;
    programs.emplace_back(std::move(prog));
    
    reset_program(programs[id]);
    return id;
  }

  void ShaderProgramManager::destroy_program(ShaderProgInternal &prog) {
    if (prog.layout)
      vkDestroyPipelineLayout(internal::app_vk_device(), prog.layout, nullptr);
    
    prog.layout = nullptr;
    prog.constants = VkPushConstantRange {0u, 0u, 0u};
    prog.valid_sets.reset();
    prog.modules.clear();
  }

  #define SPVR_ASSER(res) if ((res) != SPV_REFLECT_RESULT_SUCCESS) throw std::runtime_error {"SPVReflect error"}

  void ShaderProgramManager::reset_program(ShaderProgInternal &prog) {
    if (prog.layout)
      vkDestroyPipelineLayout(internal::app_vk_device(), prog.layout, nullptr);
    
    prog.layout = nullptr;
    prog.constants = VkPushConstantRange {0u, 0u, 0u};
    prog.valid_sets.reset();
    
    std::array<DescriptorSetLayoutInfo, MAX_DESCRIPTORS> descriptors {};
    
    for (auto mod_id : prog.modules) {
      auto &smod = get_module(mod_id);
      auto &resources = smod.get_resources();

      uint32_t count = 0;
      SPVR_ASSER(spvReflectEnumerateDescriptorSets(&resources, &count, nullptr));

      std::vector<SpvReflectDescriptorSet*> sets(count);
      SPVR_ASSER(spvReflectEnumerateDescriptorSets(&resources, &count, sets.data()));

      for (auto &set : sets) {
        if (set->set >= MAX_DESCRIPTORS)
          throw std::runtime_error {"Max set should be less then MAX_DESCRIPTORS"};
        prog.valid_sets.set(set->set);
        descriptors[set->set].parse_resources(smod.get_stage(), set);
      }

      if (resources.push_constant_block_count > 1) {
        throw std::runtime_error {"Only 1 push_const block is supported"};
      }

      if (resources.push_constant_block_count) {
        auto &blk = resources.push_constant_blocks[0];
        if (blk.offset != 0) {
          throw std::runtime_error {"PushConst offset n e 0"};
        }

        if (!prog.constants.stageFlags) {
          prog.constants.size = blk.size;
        } else if (prog.constants.size != blk.size) {
          throw std::runtime_error {"PushConst size mismatch"};
        }

        prog.constants.stageFlags |= smod.get_stage();
      }
    }

    std::vector<VkDescriptorSetLayout> vk_layouts;
    vk_layouts.reserve(MAX_DESCRIPTORS); 
    
    for (uint32_t i = 0; i < MAX_DESCRIPTORS; i++) {
      if (!prog.valid_sets.test(i))
        continue;
      auto id = cached_descriptors.register_layout(descriptors[i]);
      
      prog.sets[i] = id;
      vk_layouts.push_back(cached_descriptors.get_layout(id));
    }

    VkPipelineLayoutCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = (uint32_t)vk_layouts.size(),
      .pSetLayouts = vk_layouts.data(),
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr
    };

    if (prog.constants.stageFlags) {
      info.pushConstantRangeCount = 1;
      info.pPushConstantRanges = &prog.constants;
    }

    VKCHECK(vkCreatePipelineLayout(internal::app_vk_device(), &info, nullptr, &prog.layout));
  }

  void ShaderProgramManager::reload() {
    for (auto &prog : programs) {
      vkDestroyPipelineLayout(internal::app_vk_device(), prog.layout, nullptr);
      prog.layout = nullptr;
    }
    
    cached_descriptors.clear();

    for (auto &mod : modules) {
      mod.reload();
    }

    for (auto &prog : programs) {
      reset_program(prog);
    }
  }

  void ShaderProgramManager::clear() {
    for (auto &prog : programs) {
      vkDestroyPipelineLayout(internal::app_vk_device(), prog.layout, nullptr);
      prog.layout = nullptr;
    }

    programs.clear();
    prog_names.clear();
    cached_descriptors.clear();
    modules.clear();
    module_names.clear();
  }

  VkPipelineLayout ShaderProgramManager::get_program_layout(ShaderProgramId id) const {
    return programs.at(id).layout;
  }
  
  const std::bitset<MAX_DESCRIPTORS> &ShaderProgramManager::get_used_descriptors(ShaderProgramId id) const {
    return programs.at(id).valid_sets;
  }
  
  const DescriptorSetLayoutInfo &ShaderProgramManager::get_program_descriptor_info(ShaderProgramId id, uint32_t set) const {
    auto &prog = programs.at(id);
    if (!prog.valid_sets.test(set))
      throw std::runtime_error {"Program does not have required set"};
    return cached_descriptors.get_layout_info(prog.sets[set]); 
  }

  VkDescriptorSetLayout ShaderProgramManager::get_program_descriptor_layout(ShaderProgramId id, uint32_t set) const {
    auto &prog = programs.at(id);
    if (!prog.valid_sets.test(set))
      throw std::runtime_error {"Program does not have required set"};
    return cached_descriptors.get_layout(prog.sets[set]); 
  }

  std::vector<VkPipelineShaderStageCreateInfo> ShaderProgramManager::get_stage_info(ShaderProgramId id) const {
    auto &prog = programs.at(id);
    
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(prog.modules.size());

    for (auto mod_id : prog.modules) {
      auto &mod = modules.at(mod_id);
      
      VkPipelineShaderStageCreateInfo stage {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = mod.get_stage(),
        .module = mod.get_module(),
        .pName = mod.get_name().data(),
        .pSpecializationInfo = nullptr
      };

      stages.push_back(stage);
    }

    return stages;
  }
}