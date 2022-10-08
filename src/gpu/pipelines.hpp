#ifndef PIPELINES_HPP_INCLUDED
#define PIPELINES_HPP_INCLUDED

#include "driver.hpp"
#include "shader_program.hpp"

#include <vector>
#include <functional>
#include <string>
#include <optional>
#include <memory>

#include <lib/spirv-reflect/spirv_reflect.h>

namespace gpu {

  template <typename T> 
  struct HashFunc;

  struct RenderSubpassDesc {
    bool operator==(const RenderSubpassDesc &desc) const {
      return formats == desc.formats && use_depth == desc.use_depth;
    }

    bool use_depth = false;
    std::vector<VkFormat> formats {};
  };

  template <>
  struct HashFunc<RenderSubpassDesc> : std::binary_function<RenderSubpassDesc, RenderSubpassDesc, std::size_t> {
    std::size_t operator()(const RenderSubpassDesc &desc) const {
      std::size_t h = 0;
      for (auto fmt : desc.formats) {
        hash_combine(h, fmt);
      }
      hash_combine(h, desc.use_depth);
      return h;
    }
  };

  bool operator==(const VkVertexInputBindingDescription &l, const VkVertexInputBindingDescription &r);
  bool operator==(const VkVertexInputAttributeDescription &l, const VkVertexInputAttributeDescription &r);
  bool operator==(const VkPipelineInputAssemblyStateCreateInfo &l, const VkPipelineInputAssemblyStateCreateInfo &r);
  bool operator==(const VkPipelineRasterizationStateCreateInfo &l, const VkPipelineRasterizationStateCreateInfo &r);
  bool operator==(const VkPipelineDepthStencilStateCreateInfo &l, const VkPipelineDepthStencilStateCreateInfo &r);

  struct VertexInput {
    VertexInput() {}

    bool operator==(const VertexInput &input) const {
      bool ret = true;
      if (bindings.size() != input.bindings.size() || attributes.size() != input.attributes.size()) {
        return false;
      }
      
      for (uint32_t i = 0; i < bindings.size(); i++) {
        ret &= bindings[i] == input.bindings[i];
      }

      for (uint32_t i = 0; i < bindings.size(); i++) {
        ret &= attributes[i] == input.attributes[i];
      }

      return ret;
    }

    std::vector<VkVertexInputBindingDescription> bindings {};
    std::vector<VkVertexInputAttributeDescription> attributes {};
  };

  template <>
  struct HashFunc<VertexInput> : std::binary_function<VertexInput, VertexInput, std::size_t> {
    std::size_t operator()(const VertexInput &desc) const {
      std::size_t h = 0;
      for (auto &bind :  desc.bindings) {
        hash_combine(h, bind.binding);
        hash_combine(h, bind.inputRate);
        hash_combine(h, bind.stride);
      }

      for (auto &attr : desc.attributes) {
        hash_combine(h, attr.binding);
        hash_combine(h, attr.format);
        hash_combine(h, attr.location);
        hash_combine(h, attr.offset);
      }

      return h;
    }
  };

  struct Registers {
    
    bool operator==(const Registers &regs) const {
      return regs.assembly == assembly && regs.rasterization == rasterization && regs.depth_stencil == depth_stencil;
    }

    VkPipelineInputAssemblyStateCreateInfo assembly {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE
    };

    VkPipelineRasterizationStateCreateInfo rasterization {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .depthBiasConstantFactor = 0.f,
      .depthBiasClamp = 0.f,
      .depthBiasSlopeFactor = 0.f,
      .lineWidth = 1.f 
    };

    VkPipelineDepthStencilStateCreateInfo depth_stencil {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .depthTestEnable = VK_FALSE,
      .depthWriteEnable = VK_FALSE,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE,
      .front {},
      .back {},
      .minDepthBounds = 0.f,
      .maxDepthBounds = 1.f
    };
  };

  template <>
  struct HashFunc<Registers> : std::binary_function<Registers, Registers, std::size_t> {
    std::size_t operator()(const Registers &desc) const {
      std::size_t h = 0;
      hash_combine(h, desc.assembly.topology);
      hash_combine(h, desc.assembly.primitiveRestartEnable);

      hash_combine(h, desc.rasterization.depthClampEnable);
      hash_combine(h, desc.rasterization.rasterizerDiscardEnable);
      hash_combine(h, desc.rasterization.polygonMode);
      hash_combine(h, desc.rasterization.cullMode);
      hash_combine(h, desc.rasterization.frontFace);
      hash_combine(h, desc.rasterization.depthBiasEnable);
      hash_combine(h, desc.rasterization.depthBiasConstantFactor);
      hash_combine(h, desc.rasterization.depthBiasClamp);
      hash_combine(h, desc.rasterization.depthBiasSlopeFactor);

      hash_combine(h, desc.depth_stencil.depthTestEnable);
      hash_combine(h, desc.depth_stencil.depthWriteEnable);
      hash_combine(h, desc.depth_stencil.depthCompareOp);
      hash_combine(h, desc.depth_stencil.depthBoundsTestEnable);
      hash_combine(h, desc.depth_stencil.stencilTestEnable);
      //no front and back hashing :(
      hash_combine(h, desc.depth_stencil.depthTestEnable);
      hash_combine(h, desc.depth_stencil.minDepthBounds);
      hash_combine(h, desc.depth_stencil.maxDepthBounds);
      return h;
    }
  };

  struct PipelinePool;
  struct ProgramResources;

  struct BasePipeline {
    BasePipeline() {}
    BasePipeline(PipelinePool *base) : pool {base} {}

    void attach(PipelinePool &p) { pool = &p; }
    void set_program(const std::string &name);
    
    VkDescriptorSetLayout get_layout(uint32_t index) const;
    VkPipelineLayout get_pipeline_layout() const;
    
    bool is_attached() const { return pool != nullptr; }
    bool has_program() const { return program_id.has_value(); }

    //const ProgramResources &get_resources() const;

  protected:
    PipelinePool *pool {nullptr};
    std::optional<uint32_t> program_id {};
  };

  struct ComputePipeline : BasePipeline {
    ComputePipeline() : BasePipeline {} {}
    ComputePipeline(PipelinePool *p) : BasePipeline {p} {}

    VkPipeline get_pipeline();

    bool operator==(const ComputePipeline &p) const {
      return (pool == p.pool && p.program_id == program_id);
    }

    friend HashFunc<ComputePipeline>;
    friend PipelinePool;
  };

  struct GraphicsPipeline : BasePipeline {
    GraphicsPipeline() : BasePipeline {} {}
    GraphicsPipeline(PipelinePool *p) : BasePipeline {p} {}

    void set_vertex_input(const VertexInput &vinput);
    void set_registers(const Registers &regs);
    void set_rendersubpass(const RenderSubpassDesc &subpass);

    VkPipeline get_pipeline();
    VkRenderPass get_renderpass();
    const RenderSubpassDesc &get_renderpass_desc() const;

    bool operator==(const GraphicsPipeline &p) const {
      return pool == p.pool 
        && p.program_id == program_id
        && p.vertex_input == vertex_input
        && p.render_subpass == render_subpass
        && p.regs_index == regs_index;
    }

    bool has_vertex_input() const { return vertex_input.has_value(); }
    bool has_render_subpass() const { return render_subpass.has_value(); }
    bool has_registers() const { return regs_index.has_value(); }

  private:
    std::optional<uint32_t> vertex_input;
    std::optional<uint32_t> render_subpass;
    std::optional<uint32_t> regs_index;

    friend HashFunc<GraphicsPipeline>;
    friend PipelinePool;
  };

  template<>
  struct HashFunc<ComputePipeline> : std::binary_function<ComputePipeline, ComputePipeline, std::size_t> {
    std::size_t operator()(const ComputePipeline &p) const {
      return std::size_t(p.program_id.value());
    }
  };

  template<>
  struct HashFunc<GraphicsPipeline> : std::binary_function<GraphicsPipeline, GraphicsPipeline, std::size_t> {
    std::size_t operator()(const GraphicsPipeline &p) const {
      std::size_t h = 0;
      hash_combine(h, p.program_id.value());
      hash_combine(h, p.vertex_input.value());
      hash_combine(h, p.render_subpass.value());
      hash_combine(h, p.regs_index.value());
      return h;
    }
  };

  constexpr uint32_t BINDLESS_DESC_COUNT = 1024;

  struct PipelinePool {
    PipelinePool();
    ~PipelinePool();
    
    void create_program(const std::string &name, std::vector<std::string> &&shaders) {
      shader_programs.create_program(name, shaders);
    }

    void reload_programs();
    
    PipelinePool(const PipelinePool &) = delete;
    const PipelinePool &operator=(const PipelinePool &) = delete;
  private:
    
    struct RenderSubpass {
      RenderSubpassDesc desc;
      VkRenderPass handle = nullptr;
      bool is_empty() const { return !handle; }
      void create_renderpass();
    };

    struct Pipeline {
      VkPipeline handle = nullptr;
    };

    VkPipelineCache vk_cache {nullptr};

    ShaderProgramManager shader_programs;

    std::unordered_map<RenderSubpassDesc, uint32_t, HashFunc<RenderSubpassDesc>> render_subpasses;
    std::vector<RenderSubpass> allocated_subpasses;

    std::unordered_map<VertexInput, uint32_t, HashFunc<VertexInput>> vertex_input;
    std::vector<VertexInput> allocated_vinput;

    std::unordered_map<Registers, uint32_t, HashFunc<Registers>> registers;
    std::vector<Registers> allocated_registers;

    std::unordered_map<ComputePipeline, Pipeline, HashFunc<ComputePipeline>> compute_pipelines;
    std::unordered_map<GraphicsPipeline, Pipeline, HashFunc<GraphicsPipeline>> graphics_pipelines;

    uint32_t get_subpass_index(const RenderSubpassDesc &desc);
    VkRenderPass get_subpass(uint32_t subpass_index);
    const RenderSubpassDesc &get_subpass_desc(uint32_t subpass_index) const;

    uint32_t get_program_index(const std::string &name) const;

    uint32_t get_vinput_index(const VertexInput &vinput);
    const VertexInput &get_vinput(uint32_t index) const;

    uint32_t get_registers_index(const Registers &registers);
    const Registers &get_registers(uint32_t index) const ;

    VkPipeline get_pipeline(const ComputePipeline &pipeline);
    VkPipeline get_pipeline(const GraphicsPipeline &pipeline);
    VkRenderPass get_renderpass(const GraphicsPipeline &pipeline);

    friend BasePipeline;
    friend ComputePipeline;
    friend GraphicsPipeline;
  };
}




#endif