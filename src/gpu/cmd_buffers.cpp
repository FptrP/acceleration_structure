#include "cmd_buffers.hpp"

#include <stdexcept>
#include <iostream>

namespace gpu {

  struct FramebufferResource : CtxResource {
    FramebufferResource(VkFramebuffer framebuff) : CtxResource{},  fb {framebuff} {}

    void destroy(VkDevice device) override {
      if (device && fb) {
        vkDestroyFramebuffer(device, fb, nullptr);
      }
    }

    VkFramebuffer fb = nullptr;
  };

  EventPool::EventPool(VkDevice device, uint32_t flips_count)
    : api_device {device}, frame_count {flips_count}
  {
    allocated_events.resize(frame_count);
  }
  
  EventPool::~EventPool() {
    for (auto &pool : allocated_events) {
      for (auto &elem : pool) {
        vkDestroyEvent(api_device, elem, nullptr);
      }
    }

    for (auto &elem : used_events) {
      vkDestroyEvent(api_device, elem, nullptr);
    }
  }

  void EventPool::flip() {
    auto &pool = allocated_events[frame_index];
    for (auto elem : used_events) {
      pool.push_back(elem);
    }
    used_events.clear();
    frame_index = (frame_index + 1) % frame_count;
  }
  
  VkEvent EventPool::allocate() {
    auto &pool = allocated_events[frame_index];
    
    if (pool.size()) {
      auto result = pool.back();
      pool.pop_back();
      vkResetEvent(api_device, result);
      used_events.push_back(result);
      return result;
    }

    VkEventCreateInfo info {VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    VkEvent event;
    VKCHECK(vkCreateEvent(api_device, &info, nullptr, &event));
    used_events.push_back(event);
    return event;
  }


  void CmdContext::begin() {
    ubo_pool.reset();
    VkCommandBufferBeginInfo info {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &info);

    if (binder_state) {
      binder_state->clear();
    }
  }
  
  void CmdContext::end() {
    end_renderpass();
    
    fb_state.set_width(0);

    gfx_pipeline.reset();
    cmp_pipeline.reset();

    state.cmp_layout = nullptr;
    state.cmp_pipeline = nullptr;
    state.gfx_layout = nullptr;
    state.gfx_pipeline = nullptr;

    VKCHECK(vkEndCommandBuffer(cmd));
  }

  void CmdContext::set_framebuffer(uint32_t width, uint32_t height, const std::initializer_list<std::pair<DriverResourceID, ImageViewRange>> &attachments) {
    if (width == 0 || height == 0) {
      throw std::runtime_error {"Zero-sized framebuffer!"};
    }
    
    fb_state.set_width(width);
    fb_state.set_height(height);
    fb_state.set_layers(1);

    if (gfx_pipeline) {
      fb_state.set_renderpass(*gfx_pipeline);
    }

    for (uint32_t i = 0; i < attachments.size(); i++) {
      auto img = acquire_image(attachments.begin()[i].first);
      fb_state.set_attachment(i, img, attachments.begin()[i].second);
    }
  }

  void CmdContext::bind_pipeline(const GraphicsPipeline &pipeline) {
    if (!pipeline.is_attached()) {
      throw std::runtime_error {"Attempt to bind non-attached pipeline"};
    }
    
    bool valid = true;
    valid &= pipeline.has_program();
    valid &= pipeline.has_vertex_input();
    valid &= pipeline.has_render_subpass();
    valid &= pipeline.has_registers();

    if (!valid) {
      throw std::runtime_error {"Attemp to bind incomplite pipeline"};
    }

    gfx_pipeline = pipeline;
    
    fb_state.set_renderpass(*gfx_pipeline);

    auto renderpass = gfx_pipeline->get_renderpass();
    auto api_pipeline = gfx_pipeline->get_pipeline();

    bool reset_renderpass = (renderpass != state.renderpass) || fb_state.is_dirty();
    bool change_pipeline = api_pipeline != state.gfx_pipeline;
    
    //recreate framebuffer

    if (reset_renderpass) {
      end_renderpass();

      if (fb_state.is_dirty()) {
        flush_framebuffer_state(renderpass);
      }

      if (!state.framebuffer) {
        throw std::runtime_error {"Attempt to bind graphics pipeline without framebuffer"};
      }

      VkRenderPassBeginInfo info {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = renderpass,
        .framebuffer = state.framebuffer,
        .renderArea = {{0, 0}, {fb_state.get_width(), fb_state.get_height()}},
        .clearValueCount = 0,
        .pClearValues = nullptr
      };

      vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
      state.renderpass = renderpass;
    }

    if (change_pipeline) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, api_pipeline);
      state.gfx_pipeline = api_pipeline;
      state.gfx_layout = gfx_pipeline->get_pipeline_layout();
    }
  }
  
  void CmdContext::bind_pipeline(const ComputePipeline &pipeline) {
    if (!pipeline.is_attached()) {
      throw std::runtime_error {"Attempt to bind non-attached pipeline"};
    }

    bool valid = true;
    valid &= pipeline.has_program();

    if (!valid) {
      throw std::runtime_error {"Attemp to bind incomplite pipeline"};
    }

    cmp_pipeline = pipeline;

    state.cmp_layout = pipeline.get_pipeline_layout();
    auto api_pipeline = cmp_pipeline->get_pipeline();
    
    if (api_pipeline != state.cmp_pipeline) {
      state.cmp_pipeline = api_pipeline;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state.cmp_pipeline);
    }
  }

  void CmdContext::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance) {
    vkCmdDraw(cmd, vertex_count, instance_count, first_vertex, first_instance);
  }

  void CmdContext::draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance) {
    vkCmdDrawIndexed(cmd, index_count, instance_count, first_index, vertex_offset, first_instance);
  }
  
  void CmdContext::dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
  }

  void CmdContext::dispatch_indirect(VkBuffer buffer, VkDeviceSize offset) {
    vkCmdDispatchIndirect(cmd, buffer, offset);
  }

  void CmdContext::flush_framebuffer_state(VkRenderPass renderpass) {
    state.framebuffer = cmd_context.framebuffers.get_framebuffer(fb_state);
  }

  void CmdContext::end_renderpass() {
    if (state.renderpass) {
      vkCmdEndRenderPass(cmd);
      state.renderpass = nullptr;
    }
  }
    
  void CmdContext::bind_descriptors_compute(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets, const std::initializer_list<uint32_t> offsets) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state.cmp_layout, first_set, sets.size(), sets.begin(), offsets.size(), offsets.begin());
  }
  
  void CmdContext::bind_descriptors_graphics(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets, const std::initializer_list<uint32_t> offsets) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.gfx_layout, first_set, sets.size(), sets.begin(), offsets.size(), offsets.begin());
  }

  void CmdContext::bind_descriptors_compute(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets) {
    bind_descriptors_compute(first_set, sets, {});
  }
  
  void CmdContext::bind_descriptors_graphics(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets) {
    bind_descriptors_graphics(first_set, sets, {});
  }

  void CmdContext::bind_viewport(VkViewport viewport) {
    vkCmdSetViewport(cmd, 0, 1, &viewport);
  }
  
  void CmdContext::bind_scissors(VkRect2D scissors) {
    vkCmdSetScissor(cmd, 0, 1, &scissors);
  }

  void CmdContext::push_constants_graphics(VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void *constants) {
    vkCmdPushConstants(cmd, state.gfx_layout, stages, offset, size, constants);
  }
  
  void CmdContext::push_constants_compute(uint32_t offset, uint32_t size, const void *constants) {
    vkCmdPushConstants(cmd, state.cmp_layout, VK_SHADER_STAGE_COMPUTE_BIT, offset, size, constants);
  }

  void CmdContext::update_buffer(VkBuffer target, VkDeviceSize offset, VkDeviceSize data_size, const void *src) {
    vkCmdUpdateBuffer(cmd, target, offset, data_size, src);
  }

  void CmdContext::clear_resources() {
    if (state.framebuffer) {
      state.framebuffer = nullptr;
    }

    for (auto res : delayed_free) {
      res->destroy(internal::app_vk_device());
      delete res;
    }
    delayed_free.clear();
  }

  void CmdContext::clear_color_attachments(float r, float g, float b, float a) {
    const auto &desc = gfx_pipeline->get_renderpass_desc();
    uint32_t count = desc.formats.size();
    if (desc.use_depth) {
      count--;
    }

    VkClearAttachment clear {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .colorAttachment = 0,
      .clearValue {.color {r, g, b, a}}
    };

    VkClearRect clear_rect {
      .rect {{0, 0}, {fb_state.get_width(), fb_state.get_height()}},
      .baseArrayLayer = 0,
      .layerCount = 1
    };

    for (uint32_t i = 0; i < count; i++) {
      clear.colorAttachment = i;
      vkCmdClearAttachments(cmd, 1, &clear, 1, &clear_rect);
    }
  }

  void CmdContext::clear_depth_attachment(float val) {
    const auto &desc = gfx_pipeline->get_renderpass_desc();

    if (!desc.use_depth) {
      return;
    }

    uint32_t index = desc.formats.size() - 1;

    VkClearAttachment clear {
      .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
      .colorAttachment = index,
      .clearValue {.depthStencil {val, 0u}}
    };

    VkClearRect clear_rect {
      .rect {{0, 0}, {fb_state.get_width(), fb_state.get_height()}},
      .baseArrayLayer = 0,
      .layerCount = 1
    };
    vkCmdClearAttachments(cmd, 1, &clear, 1, &clear_rect);
  }

  void CmdContext::bind_vertex_buffers(uint32_t first_binding, const std::initializer_list<VkBuffer> &buffers, const std::initializer_list<uint64_t> &offsets) {
    vkCmdBindVertexBuffers(cmd, first_binding, buffers.size(), buffers.begin(), offsets.begin());
  }

  void CmdContext::bind_index_buffer(VkBuffer buffer, uint64_t offset, VkIndexType type) {
    vkCmdBindIndexBuffer(cmd, buffer, offset, type);
  }

  void CmdContext::signal_event(VkEvent event, VkPipelineStageFlags stages) {
    vkCmdSetEvent(cmd, event, stages);
  }

  void CmdContext::push_label(const char *name) {
    VkDebugUtilsLabelEXT label {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pNext = nullptr,
        .pLabelName = name,
        .color {0.f, 0.f, 0.f, 0.f}
      };

    vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
  }
  
  void CmdContext::pop_label() {
    vkCmdEndDebugUtilsLabelEXT(cmd);
  }

  TransferCmdPool::TransferCmdPool()
  {
    auto qinfo = app_main_queue();
    auto api_device = internal::app_vk_device();

    VkCommandPoolCreateInfo pool_info {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT|VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qinfo.family
    };

    VKCHECK(vkCreateCommandPool(api_device, &pool_info, nullptr, &pool));

    VkCommandBufferAllocateInfo alloc_info {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1
    };

    VKCHECK(vkAllocateCommandBuffers(api_device, &alloc_info, &cmd));

    VkFenceCreateInfo fence_info {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(api_device, &fence_info, nullptr, &fence));

  }

  TransferCmdPool::TransferCmdPool(TransferCmdPool &&tpool)
    : pool {tpool.pool}, cmd {tpool.cmd}, fence {tpool.fence},
      buffer_acquired {tpool.buffer_acquired}
  {
  }

  TransferCmdPool::~TransferCmdPool() {
    auto api_device = internal::app_vk_device();
    if (fence) {
      vkDestroyFence(api_device, fence, nullptr);
    }

    if (pool) {
      vkDestroyCommandPool(api_device, pool, nullptr);
    }
  }

  VkCommandBuffer TransferCmdPool::get_cmd_buffer() {
    if (buffer_acquired) {
      throw std::runtime_error {"Buffer is already acquired"};
    }
    buffer_acquired = true;
    
    return cmd;
  }
  void TransferCmdPool::submit_and_wait() {
    auto api_device = internal::app_vk_device();
    auto api_queue = app_main_queue().queue;

    if (!buffer_acquired) {
      return;
    }

    VkSubmitInfo submit_info {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 0,
      .pWaitSemaphores = nullptr,
      .pWaitDstStageMask = nullptr,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 0,
      .pSignalSemaphores = nullptr
    };

    VKCHECK(vkQueueSubmit(api_queue, 1, &submit_info, fence));
    VKCHECK(vkWaitForFences(api_device, 1, &fence, VK_TRUE, UINT64_MAX));
    VKCHECK(vkResetFences(api_device, 1, &fence));
    VKCHECK(vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));

    buffer_acquired = false;
  }

  const TransferCmdPool &TransferCmdPool::operator=(TransferCmdPool &&tpool) {
    std::swap(pool, tpool.pool);
    std::swap(cmd, tpool.cmd);
    std::swap(fence, tpool.fence);
    std::swap(buffer_acquired, tpool.buffer_acquired);
    return *this;
  }

  CmdContext::CmdContext(CmdContext &&o) :
    cmd_context {o.cmd_context},
    cmd {o.cmd},
    gfx_pipeline {o.gfx_pipeline},
    cmp_pipeline {o.cmp_pipeline},
    state {o.state},
    fb_state {std::move(o.fb_state)},
    ubo_pool {std::move(o.ubo_pool)},
    delayed_free {std::move(o.delayed_free)}
  {
    o.state.framebuffer = nullptr;
    o.cmd = nullptr;
  }

  CmdContext &CmdContext::operator=(CmdContext &&o) {
    std::swap(cmd, o.cmd);
    std::swap(gfx_pipeline, o.gfx_pipeline);
    std::swap(cmp_pipeline, o.cmp_pipeline);
    std::swap(state, o.state);
    std::swap(fb_state, o.fb_state);
    ubo_pool = std::move(o.ubo_pool);
    return *this;
  }

}