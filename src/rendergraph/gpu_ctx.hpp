#ifndef RENDERGRAPH_CPU_CTX_HPP_INCLUDED
#define RENDERGRAPH_CPU_CTX_HPP_INCLUDED

#include "gpu/driver.hpp"

#include "resources.hpp"

namespace rendergraph {

  struct GpuState {
    GpuState()
      : backbuffers_count { gpu::app_swapchain().get_images_count()},
        frames_count {backbuffers_count},
        desc_pool {frames_count},
        event_pool {gpu::app_device().api_device(), frames_count},
        ctx_pool {frames_count}
    {
      submit_fences.reserve(frames_count);
      image_acquire_semaphores.reserve(frames_count);
      submit_done_semaphores.reserve(frames_count);

      for (uint32_t i = 0; i < frames_count; i++) {
        submit_fences.push_back(gpu::Fence(true));
      }

      for (uint32_t i = 0; i < backbuffers_count; i++) {
        image_acquire_semaphores.push_back({});
        submit_done_semaphores.push_back({});
      }
    }

    ~GpuState() { vkDeviceWaitIdle(gpu::app_device().api_device()); }

    void acquire_image();
    void begin();
    void submit(bool present);

    gpu::CmdContext &get_cmdbuff() { return ctx_pool.get_ctx(); }
    
    uint32_t get_frame_index() const { return frame_index; }
    uint32_t get_backbuf_index() const { return backbuf_index; }

    std::vector<gpu::ImagePtr> take_backbuffers() { return gpu::get_swapchain_image_ptr(); }

    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) { return desc_pool.allocate_set(layout); }
    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout, const std::vector<uint32_t> &variable_sizes) { return desc_pool.allocate_set(layout, variable_sizes); }
    
    VkEvent allocate_event() { return event_pool.allocate(); }
    
    uint32_t get_frames_count() const { return frames_count; }
    
    uint32_t get_backbuffers_count() const { return backbuffers_count;}
  private:
    uint32_t backbuffers_count = 0;
    uint32_t frames_count = 0;

    //gpu::CmdBufferPool cmdbuffer_pool;
    gpu::DescriptorPool desc_pool;
    gpu::EventPool event_pool;

    //std::vector<gpu::CmdContext> cmd_buffers;
    gpu::CmdContextPool ctx_pool;
    std::vector<gpu::Fence> submit_fences;
    std::vector<gpu::Semaphore> image_acquire_semaphores;
    std::vector<gpu::Semaphore> submit_done_semaphores;  

    
    uint32_t frame_index = 0;
    uint32_t backbuf_index = 0;
    uint32_t backbuf_sem_index = 0;
  };

}

#endif