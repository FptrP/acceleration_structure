#include "pipelines.hpp"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

#include <lib/spirv-reflect/spirv_reflect.h>

#include <sstream>
#include <initializer_list>
#include <map>

namespace gpu {

  PipelinePool::PipelinePool() {
    VkPipelineCacheCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .initialDataSize = 0,
      .pInitialData = nullptr
    };
    VKCHECK(vkCreatePipelineCache(internal::app_vk_device(), &info, nullptr, &vk_cache));
  }

  PipelinePool::~PipelinePool() {
    for (auto &[k, v] : compute_pipelines) {
      vkDestroyPipeline(internal::app_vk_device(), v.handle, nullptr);
    }

    for (auto &[k, v] : graphics_pipelines) {
      vkDestroyPipeline(internal::app_vk_device(), v.handle, nullptr);
    }

    if (vk_cache) {
      vkDestroyPipelineCache(internal::app_vk_device(), vk_cache, nullptr);
    }

    for (auto &subpass : allocated_subpasses) {
      if (!subpass.is_empty()) {
        vkDestroyRenderPass(internal::app_vk_device(), subpass.handle, nullptr);
      } 
    }
  }


  void PipelinePool::reload_programs() {
    for (auto &desc : compute_pipelines) {
      vkDestroyPipeline(internal::app_vk_device(), desc.second.handle, nullptr);
      desc.second.handle = nullptr;
    }

    for (auto &desc : graphics_pipelines) {
      vkDestroyPipeline(internal::app_vk_device(), desc.second.handle, nullptr);
      desc.second.handle = nullptr;
    }
    
    shader_programs.reload();
  }

  uint32_t PipelinePool::get_subpass_index(const RenderSubpassDesc &desc) {
    if (!render_subpasses.count(desc)) {
      uint32_t index = allocated_subpasses.size();
      allocated_subpasses.push_back({desc, nullptr});
      render_subpasses[desc] = index;
      return index;
    }

    return render_subpasses[desc];
  }
  
  void PipelinePool::RenderSubpass::create_renderpass() {
    std::vector<VkAttachmentDescription> attachments;
    
    VkAttachmentDescription attach_desc {
      0,
      VK_FORMAT_R8G8B8A8_SRGB,
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_LOAD,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };
    
    for (auto fmt : desc.formats) {
      attach_desc.format = fmt;
      attachments.push_back(attach_desc);
    }

    if (desc.use_depth) {
      attachments.back().initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      attachments.back().finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    std::vector<VkAttachmentReference> references;
    for (uint32_t i = 0; i < attachments.size(); i++) {
      references.push_back({i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    }

    if (desc.use_depth) {
      references.back().layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkSubpassDescription subpass {};
    subpass.colorAttachmentCount = desc.use_depth? (references.size() - 1) : references.size();
    subpass.pColorAttachments = references.data();
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (desc.use_depth) {
      subpass.pDepthStencilAttachment = &references.back();
    }

    VkRenderPassCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = attachments.size();
    info.pAttachments = attachments.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    
    VKCHECK(vkCreateRenderPass(internal::app_vk_device(), &info, nullptr, &handle));
  }

  VkRenderPass PipelinePool::get_subpass(uint32_t subpass_index) {
    auto &subpass = allocated_subpasses.at(subpass_index);
    if (subpass.is_empty()) {
      subpass.create_renderpass();
    }
    return subpass.handle;
  }

  const RenderSubpassDesc &PipelinePool::get_subpass_desc(uint32_t subpass_index) const {
    auto &subpass = allocated_subpasses.at(subpass_index);
    return subpass.desc;
  }

  uint32_t PipelinePool::get_program_index(const std::string &name) const {
    return shader_programs.get_program(name);
  }

  uint32_t PipelinePool::get_vinput_index(const VertexInput &vinput) {
    auto it = vertex_input.find(vinput);
    if (it == vertex_input.end()) {
      uint32_t index = allocated_vinput.size();
      allocated_vinput.push_back(vinput);
      vertex_input.insert({vinput, index});
      return index;
    }

    return it->second;
  }
  
  const VertexInput &PipelinePool::get_vinput(uint32_t index) const {
    return allocated_vinput.at(index);
  }

  uint32_t PipelinePool::get_registers_index(const Registers &regs) {
    auto it = registers.find(regs);
    
    if (it == registers.end()) {
      uint32_t index = allocated_registers.size();
      allocated_registers.push_back(regs);
      registers.insert({regs, index});
      return index;
    }

    return it->second;
  }

  const Registers &PipelinePool::get_registers(uint32_t index) const {
    return allocated_registers.at(index);
  }

  void BasePipeline::set_program(const std::string &name) {
    program_id = pool->get_program_index(name);
  }

  VkDescriptorSetLayout BasePipeline::get_layout(uint32_t index) const {
    if (!program_id.has_value()) {
      throw std::runtime_error {"Pipeline not attached to program"};
    }
    
    return pool->shader_programs.get_program_descriptor_layout(program_id.value(), index);
  }
  
  VkPipelineLayout BasePipeline::get_pipeline_layout() const {
    if (!program_id.has_value()) {
      throw std::runtime_error {"Pipeline not attached to program"};
    }
    return pool->shader_programs.get_program_layout(program_id.value());
  }

  VkPipeline PipelinePool::get_pipeline(const ComputePipeline &pipeline) {
    auto &res = compute_pipelines[pipeline];
    if (res.handle) {
      return res.handle;
    }

    auto stages = shader_programs.get_stage_info(pipeline.program_id.value());
    if (stages.size() != 1 || stages[0].stage != VK_SHADER_STAGE_COMPUTE_BIT) {
      throw std::runtime_error {"Not compute program"};
    }

    VkComputePipelineCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = stages[0],
      .layout = shader_programs.get_program_layout(pipeline.program_id.value()),
      .basePipelineHandle = nullptr,
      .basePipelineIndex = 0
    };

    VKCHECK(vkCreateComputePipelines(internal::app_vk_device(), vk_cache, 1, &info, nullptr, &res.handle));
    return res.handle;
  }

  VkPipeline PipelinePool::get_pipeline(const GraphicsPipeline &pipeline) {
    auto &res = graphics_pipelines[pipeline];
    if (res.handle) {
      return res.handle;
    }

    const auto &regs = get_registers(pipeline.regs_index.value());
    const auto &vinput = get_vinput(pipeline.vertex_input.value());
    auto renderpass = get_renderpass(pipeline);
    const auto &rp_desc = get_subpass_desc(pipeline.render_subpass.value());

    auto stages = shader_programs.get_stage_info(pipeline.program_id.value());

    VkPipelineColorBlendAttachmentState blend_attachment {
      .blendEnable = VK_FALSE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT
    };

    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments;
    uint32_t att_count = rp_desc.use_depth? (rp_desc.formats.size() - 1) : rp_desc.formats.size();
    blend_attachments.insert(blend_attachments.begin(), att_count, blend_attachment);

    VkPipelineColorBlendStateCreateInfo blend_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_CLEAR,
      .attachmentCount = (uint32_t)blend_attachments.size(),
      .pAttachments = blend_attachments.data(),
      .blendConstants {1.f, 1.f, 1.f, 1.f}
    };
    
    VkDynamicState dyn_states[] {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .dynamicStateCount = 2,
      .pDynamicStates = dyn_states
    };
    
    VkPipelineVertexInputStateCreateInfo vertex_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .vertexBindingDescriptionCount = (uint32_t)vinput.bindings.size(),
      .pVertexBindingDescriptions = vinput.bindings.data(),
      .vertexAttributeDescriptionCount = (uint32_t)vinput.attributes.size(),
      .pVertexAttributeDescriptions = vinput.attributes.data()
    };

    VkViewport empty_vp {0.f, 0.f, 1.f, 1.f, 0.f, 1.f};
    VkRect2D empty_scissor {{0, 0}, {1, 1}};

    VkPipelineViewportStateCreateInfo viewport_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .viewportCount = 1,
      .pViewports = &empty_vp,
      .scissorCount = 1,
      .pScissors = &empty_scissor
    };

    VkPipelineMultisampleStateCreateInfo multisample {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .pNext = 0,
      .flags = 0,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
      .minSampleShading = 0.f,
      .pSampleMask = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE
    };

    VkGraphicsPipelineCreateInfo info {};

    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.stageCount = (uint32_t)stages.size();
    info.pStages = stages.data();
    info.pVertexInputState = &vertex_state;
    info.pInputAssemblyState = &regs.assembly;
    info.pRasterizationState = &regs.rasterization;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &regs.depth_stencil;
    info.pColorBlendState = &blend_state;
    info.pDynamicState = &dynamic_state;
    info.pViewportState = &viewport_state;
    info.pTessellationState = nullptr;
    info.renderPass = renderpass;
    info.subpass = 0;
    info.layout = shader_programs.get_program_layout(pipeline.program_id.value());

    VKCHECK(vkCreateGraphicsPipelines(internal::app_vk_device(), vk_cache, 1, &info, nullptr, &res.handle));

    return res.handle;
  }
  
  VkRenderPass PipelinePool::get_renderpass(const GraphicsPipeline &pipeline) {
    return get_subpass(pipeline.render_subpass.value());
  }

  VkPipeline ComputePipeline::get_pipeline() {
    return pool->get_pipeline(*this);
  }

  VkPipeline GraphicsPipeline::get_pipeline() {
    return pool->get_pipeline(*this);
  }
  
  VkRenderPass GraphicsPipeline::get_renderpass() {
    return pool->get_renderpass(*this);
  }

  const RenderSubpassDesc &GraphicsPipeline::get_renderpass_desc() const {
    return pool->get_subpass_desc(render_subpass.value());
  }

  void GraphicsPipeline::set_vertex_input(const VertexInput &vinput) {
    vertex_input = pool->get_vinput_index(vinput);
  }
  
  void GraphicsPipeline::set_registers(const Registers &regs) {
    regs_index = pool->get_registers_index(regs);
  }
  
  void GraphicsPipeline::set_rendersubpass(const RenderSubpassDesc &subpass) {
    render_subpass = pool->get_subpass_index(subpass);
  }


  bool operator==(const VkVertexInputBindingDescription &l, const VkVertexInputBindingDescription &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }

  bool operator==(const VkVertexInputAttributeDescription &l, const VkVertexInputAttributeDescription &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }
  
  bool operator==(const VkPipelineInputAssemblyStateCreateInfo &l, const VkPipelineInputAssemblyStateCreateInfo &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }

  bool operator==(const VkPipelineRasterizationStateCreateInfo &l, const VkPipelineRasterizationStateCreateInfo &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }
  
  bool operator==(const VkPipelineDepthStencilStateCreateInfo &l, const VkPipelineDepthStencilStateCreateInfo &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }

}