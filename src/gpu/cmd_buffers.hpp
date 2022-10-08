#ifndef BUFFER_POOL_HPP_INCLUDED
#define BUFFER_POOL_HPP_INCLUDED

#include <vector>

#include "driver.hpp"

#include <vector>
#include <algorithm>
#include <memory>

#include "gpu/pipelines.hpp"
#include "gpu/dynbuffer.hpp"
#include "descriptors.hpp"
#include "framebuffers.hpp"

namespace gpu {

  struct CmdContext;

  struct CmdBufferPool {
    CmdBufferPool()
    {
      auto device = internal::app_vk_device();
      auto qinfo = app_main_queue();

      VkCommandPoolCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = qinfo.family
      };

      VKCHECK(vkCreateCommandPool(device, &info, nullptr, &pool));
    }
    
    ~CmdBufferPool() {
      if (pool) {
        vkDestroyCommandPool(internal::app_vk_device(), pool, nullptr);
      }
    }

    CmdBufferPool(CmdBufferPool &&o) : pool {o.pool} { o.pool = nullptr; }
    
    const CmdBufferPool &operator=(CmdBufferPool &&o) {
      std::swap(pool, o.pool);
      return *this;
    }

    std::vector<VkCommandBuffer> allocate(uint32_t count) {
      VkCommandBufferAllocateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = count
      };

      std::vector<VkCommandBuffer> result;
      result.resize(count);
      VKCHECK(vkAllocateCommandBuffers(internal::app_vk_device(), &info, result.data()));
      return result;
    }

    VkCommandBuffer allocate() {
      VkCommandBufferAllocateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
      };

      VkCommandBuffer result;
      VKCHECK(vkAllocateCommandBuffers(internal::app_vk_device(), &info, &result));
      return result;
    }

  private:
    VkCommandPool pool {VK_NULL_HANDLE};

    CmdBufferPool(const CmdBufferPool&) = delete;
    const CmdBufferPool &operator=(const CmdBufferPool&) = delete; 
  };


  struct CtxResource {
    CtxResource() {}
    virtual void destroy(VkDevice device) = 0;
    virtual ~CtxResource() {} 
  };

  struct EventPool {
    EventPool(VkDevice device, uint32_t flips_count);
    ~EventPool();

    void flip();
    VkEvent allocate();

  private:
    VkDevice api_device = nullptr;
    uint32_t frame_index = 0;
    uint32_t frame_count = 0;

    std::vector<std::vector<VkEvent>> allocated_events;
    std::vector<VkEvent> used_events;
  };

  struct CmdMarker {
    CmdMarker(VkCommandBuffer target, const char *name) : cmd {target} {
      VkDebugUtilsLabelEXT label {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pNext = nullptr,
        .pLabelName = name,
        .color {0.f, 0.f, 0.f, 0.f}
      };

      vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
    }

    ~CmdMarker() {
      vkCmdEndDebugUtilsLabelEXT(cmd);
    }

  private:
    VkCommandBuffer cmd;
  };

  constexpr uint64_t UBO_POOL_SIZE = 16 * (1 << 10); //16Kb

  struct CmdContextPool {
    CmdContextPool(uint32_t num_frames)
      : framebuffers {FRAMES_TO_COLLECT}
    {
      auto cmd_buffers = pool.allocate(num_frames);
      ctx.reserve(num_frames);
      for (uint32_t i = 0; i < num_frames; i++) {
        ctx.emplace_back(*this, cmd_buffers[i], app_device().get_properties().limits.minUniformBufferOffsetAlignment);  
      }
    }

    ~CmdContextPool() {}

    void flip() {
      framebuffers.flip();
      ctx_index = (ctx_index + 1) % ctx.size();
    }
    
    CmdContext &get_ctx() { return ctx[ctx_index]; } 

  private:
    CmdBufferPool pool;
    FramebuffersCache framebuffers;

    uint32_t ctx_index = 0;
    std::vector<CmdContext> ctx;

    static constexpr uint32_t FRAMES_TO_COLLECT = 10;
    friend CmdContext;
  };

  struct CmdContext {
    CmdContext(CmdContextPool &base, VkCommandBuffer cmd_buf, uint64_t alignment)
      : cmd_context {base}, cmd {cmd_buf}, ubo_pool {alignment, UBO_POOL_SIZE} {}
    ~CmdContext() { clear_resources(); }
    
    void begin();
    void end();

    void set_framebuffer(uint32_t width, uint32_t height, const std::initializer_list<std::pair<DriverResourceID, ImageViewRange>> &attachments);

    void bind_pipeline(const GraphicsPipeline &pipeline);
    void bind_pipeline(const ComputePipeline &pipeline);
    void end_renderpass();

    void bind_descriptors_compute(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets, const std::initializer_list<uint32_t> offsets);
    void bind_descriptors_graphics(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets, const std::initializer_list<uint32_t> offsets);
    void bind_descriptors_compute(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets);
    void bind_descriptors_graphics(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets);

    void bind_viewport(VkViewport viewport);
    void bind_scissors(VkRect2D scissors);
    void bind_viewport(float x, float y, float w, float h, float min_d, float max_d) { bind_viewport({x, y, w, h, min_d, max_d}); }
    void bind_scissors(int32_t x, int32_t y, uint32_t w, uint32_t h) { bind_scissors({{x, y}, {w, h}}); }
    
    void bind_vertex_buffers(uint32_t first_binding, const std::initializer_list<VkBuffer> &buffers, const std::initializer_list<uint64_t> &offsets);
    void bind_index_buffer(VkBuffer buffer, uint64_t offset, VkIndexType type);

    void clear_color_attachments(float r, float g, float b, float a);
    void clear_depth_attachment(float val);

    void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance);
    void draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance);
    void dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
    void dispatch_indirect(VkBuffer buffer, VkDeviceSize offset = 0);

    void push_constants_graphics(VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void *constants);
    void push_constants_compute(uint32_t offset, uint32_t size, const void *constants);
    
    void update_buffer(VkBuffer target, VkDeviceSize offset, VkDeviceSize data_size, const void *src);

    template <typename T>
    void update_buffer(VkBuffer target, VkDeviceSize offset, const T &data) {
      update_buffer(target, offset, sizeof(data), &data);
    }

    void push_label(const char *name);
    void pop_label();
    
    void signal_event(VkEvent event, VkPipelineStageFlags stages);

    VkCommandBuffer get_command_buffer() const { return cmd; }
    void clear_resources();

    UniformBufferPool &get_ubo_pool() { return ubo_pool; }

    template<typename T>
    UboBlock<T> allocate_ubo() { return ubo_pool.allocate_ubo<T>(); }

    CmdContext(CmdContext &&) /*= default*/;
    CmdContext &operator=(CmdContext &&) /*= default*/;
    //void draw_indexed
  private:
    CmdContextPool &cmd_context;
    VkCommandBuffer cmd = nullptr;

    std::optional<GraphicsPipeline> gfx_pipeline {};
    std::optional<ComputePipeline> cmp_pipeline {};

    struct BindedState {
      VkRenderPass renderpass = nullptr;
      VkFramebuffer framebuffer = nullptr;
      VkPipeline gfx_pipeline = nullptr;
      VkPipelineLayout gfx_layout = nullptr;
      VkPipeline cmp_pipeline = nullptr;
      VkPipelineLayout cmp_layout = nullptr;
    } state {};

    FramebufferState fb_state;

    UniformBufferPool ubo_pool;

    std::vector<CtxResource*> delayed_free {};
    std::shared_ptr<DescriptorBinder> binder_state; 

    void flush_framebuffer_state(VkRenderPass renderpass);
  };

  struct TransferCmdPool {
    TransferCmdPool();
    TransferCmdPool(TransferCmdPool &&pool);
    ~TransferCmdPool();

    VkCommandBuffer get_cmd_buffer();
    void submit_and_wait();

    const TransferCmdPool &operator=(TransferCmdPool &&pool);

  private:
    VkCommandPool pool {nullptr};
    VkCommandBuffer cmd {nullptr};
    VkFence fence {nullptr};
    bool buffer_acquired = false;
    
    TransferCmdPool(TransferCmdPool &) = delete;
    const TransferCmdPool &operator=(const TransferCmdPool &pool);
  };

}

#endif