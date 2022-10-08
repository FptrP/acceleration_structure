#include "gpu_ctx.hpp"

namespace rendergraph {

  void GpuState::acquire_image() {
    VKCHECK(vkAcquireNextImageKHR(
      gpu::app_device().api_device(),
      gpu::app_swapchain().api_swapchain(),
      ~0ull,
      image_acquire_semaphores[backbuf_sem_index],
      nullptr, &backbuf_index));
  }

  void GpuState::begin() {

    auto &cmd = ctx_pool.get_ctx(); 
    VkFence cmd_fence = submit_fences[frame_index];

    vkWaitForFences(gpu::app_device().api_device(), 1, &cmd_fence, VK_TRUE, UINT64_MAX);
    submit_fences[frame_index].reset();
    vkResetCommandBuffer(cmd.get_command_buffer(), VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    desc_pool.flip();
    event_pool.flip();

    cmd.begin();
    cmd.clear_resources();
  }
  
  void GpuState::submit(bool present) {
    if (!present) {
      auto &cmd = ctx_pool.get_ctx();
      auto api_cmd = cmd.get_command_buffer();
      VkFence cmd_fence = submit_fences[frame_index];
      auto queue = gpu::app_device().api_queue();

      cmd.end();

      VkSubmitInfo submit_info {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &api_cmd,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr
      };

      VKCHECK(vkQueueSubmit(queue, 1, &submit_info, cmd_fence));
      frame_index = (frame_index + 1) % frames_count;
      ctx_pool.flip();
      return;
    }


    auto &cmd = ctx_pool.get_ctx();
    auto api_cmd = cmd.get_command_buffer();
    VkFence cmd_fence = submit_fences[frame_index];
    
    auto api_swapchain = gpu::app_swapchain().api_swapchain();
    auto queue = gpu::app_device().api_queue();

    cmd.end();

    VkSemaphore wait_sem = image_acquire_semaphores[backbuf_sem_index];
    VkSemaphore signal_sem = submit_done_semaphores[backbuf_sem_index];
    VkPipelineStageFlags wait_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkSubmitInfo submit_info {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &wait_sem,
      .pWaitDstStageMask = &wait_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &api_cmd,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &signal_sem
    };

    VKCHECK(vkQueueSubmit(queue, 1, &submit_info, cmd_fence));

    VkResult present_result;

    VkPresentInfoKHR present_info {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &signal_sem,
      .swapchainCount = 1,
      .pSwapchains = &api_swapchain,
      .pImageIndices = &backbuf_index,
      .pResults = &present_result
    };

    VKCHECK(vkQueuePresentKHR(queue, &present_info));
    VKCHECK(present_result);

    frame_index = (frame_index + 1) % frames_count;
    backbuf_sem_index = (backbuf_sem_index + 1) % backbuffers_count;
    ctx_pool.flip();
    acquire_image();
  }

}