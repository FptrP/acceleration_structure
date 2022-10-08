#include "rendergraph.hpp"
#include <iostream>

namespace rendergraph {
  
  static VkPipelineStageFlags get_pipeline_flags(VkShaderStageFlags stages) {
    VkPipelineStageFlags pipeline_stages = 0;
    if (stages & VK_SHADER_STAGE_VERTEX_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_COMPUTE_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    return pipeline_stages;
  }

  ImageViewId RenderGraphBuilder::use_color_attachment(ImageResourceId id, uint32_t mip, uint32_t layer) {
    ImageSubresourceId subres {id, mip, layer};
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };
    tracking_state.add_input(resources, subres, state);
    return ImageViewId {id, gpu::ImageViewRange {VK_IMAGE_VIEW_TYPE_2D, mip, 1, layer, 1}};
  }
  
  ImageViewId RenderGraphBuilder::use_depth_attachment(ImageResourceId id, uint32_t mip, uint32_t layer) {
    ImageSubresourceId subres {id, mip, layer};
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    };
    
    tracking_state.add_input(resources, subres, state);
    return ImageViewId {id, gpu::ImageViewRange {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, mip, 1, layer, 1}};
  }
  
  ImageViewId RenderGraphBuilder::use_storage_image(ImageResourceId id, VkShaderStageFlags stages, uint32_t mip, uint32_t layer) {
    auto pipeline_stages = get_pipeline_flags(stages);
    ImageSubresourceId subres {id, mip, layer};
    ImageSubresourceState state {
      pipeline_stages,
      VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
      VK_IMAGE_LAYOUT_GENERAL
    };
    
    tracking_state.add_input(resources, subres, state);
    return ImageViewId {id, gpu::ImageViewRange {VK_IMAGE_VIEW_TYPE_2D, mip, 1, layer, 1}};
  }

  ImageViewId RenderGraphBuilder::use_storage_image_array(ImageResourceId id, VkShaderStageFlags stages) {
    auto pipeline_stages = get_pipeline_flags(stages);
    const auto &desc = resources.get_info(id);
    
    ImageSubresourceState state {
      pipeline_stages,
      VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
      VK_IMAGE_LAYOUT_GENERAL
    };

    for (uint32_t layer = 0; layer < desc.arrayLayers; layer++) {
      ImageSubresourceId subres {id, 0, layer};
      tracking_state.add_input(resources, subres, state);
    }
    
    return ImageViewId {id, gpu::ImageViewRange {VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, 1, 0, desc.arrayLayers}};
  }

  ImageViewId RenderGraphBuilder::sample_image(ImageResourceId id, VkShaderStageFlags stages, VkImageAspectFlags aspect) {
    const auto &desc = resources.get_info(id);
    return sample_image(id, stages, aspect, 0, desc.mipLevels, 0, desc.arrayLayers);
  }

  ImageViewId RenderGraphBuilder::sample_image(ImageResourceId id, VkShaderStageFlags stages, VkImageAspectFlags aspect, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count) {
    auto pipeline_stages = get_pipeline_flags(stages);

    ImageSubresourceState state {
      pipeline_stages,
      VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };

    for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
      for (uint32_t mip = base_mip; mip < base_mip + mip_count; mip++) {
        ImageSubresourceId subres {id, mip, layer};
        tracking_state.add_input(resources, subres, state);
      }
    }
    auto type = (layer_count > 1)? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; 
    return ImageViewId {id, {type, aspect, base_mip, mip_count, base_layer, layer_count}};
  }

  ImageViewId RenderGraphBuilder::sample_cubemap(ImageResourceId id, VkShaderStageFlags stages, VkImageAspectFlags aspect) {
    const auto &desc = resources.get_info(id);
    auto pipeline_stages = get_pipeline_flags(stages);

    ImageSubresourceState state {
      pipeline_stages,
      VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };

    for (uint32_t layer = 0; layer < desc.arrayLayers; layer++) {
      for (uint32_t mip = 0; mip < desc.mipLevels; mip++) {
        ImageSubresourceId subres {id, mip, layer};
        tracking_state.add_input(resources, subres, state);
      }
    }

    return ImageViewId {id, {VK_IMAGE_VIEW_TYPE_CUBE, aspect, 0, desc.mipLevels, 0, desc.arrayLayers}};
  }

  void RenderGraphBuilder::transfer_read(ImageResourceId id, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count) {
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_READ_BIT,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    };

    for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
      for (uint32_t mip = base_mip; mip < base_mip + mip_count; mip++) {
        ImageSubresourceId subres {id, mip, layer};
        tracking_state.add_input(resources, subres, state);
      }
    }

  }
  
  void RenderGraphBuilder::transfer_write(ImageResourceId id, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count) {
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    };

    for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
      for (uint32_t mip = base_mip; mip < base_mip + mip_count; mip++) {
        ImageSubresourceId subres {id, mip, layer};
        tracking_state.add_input(resources, subres, state);
      }
    }
  }

  void RenderGraphBuilder::transfer_write(BufferResourceId id) {
    BufferState state {
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT
    };
    tracking_state.add_input(resources, id, state);
  }

  void RenderGraphBuilder::use_uniform_buffer(BufferResourceId id, VkShaderStageFlags stages) {
    auto pipeline_stages = get_pipeline_flags(stages);
    tracking_state.add_input(resources, id, {pipeline_stages, VK_ACCESS_UNIFORM_READ_BIT});
  }

  void RenderGraphBuilder::use_storage_buffer(BufferResourceId id, VkShaderStageFlags stages, bool readonly) {
    auto pipeline_stages = get_pipeline_flags(stages);
    VkAccessFlags access = VK_ACCESS_SHADER_READ_BIT;
    
    if (!readonly) {
      access |= VK_ACCESS_SHADER_WRITE_BIT;
    }

    tracking_state.add_input(resources, id, {pipeline_stages, access});
  }

  void RenderGraphBuilder::use_indirect_buffer(BufferResourceId id) {
    VkPipelineStageFlags pipeline_stages = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    VkAccessFlags access = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    tracking_state.add_input(resources, id, BufferState {pipeline_stages, access});
  }

  void RenderGraphBuilder::prepare_backbuffer() {
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      0,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };

    ImageSubresourceId subres {backbuffer, 0, 0};
    tracking_state.add_input(resources, subres, state);
    present_backbuffer = true;
  }

  ImageViewId RenderGraphBuilder::use_backbuffer_attachment() {

    ImageSubresourceState state {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    ImageSubresourceId subres {backbuffer, 0, 0};
    tracking_state.add_input(resources, subres, state);
    return {backbuffer, {VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1}};
  }

  
  gpu::ImageInfo RenderGraphBuilder::get_image_info(ImageResourceId id) {
    auto info = resources.get_info(id);
    gpu::ImageInfo res {};
    
    res.width = info.extent.width;
    res.height = info.extent.height;
    res.depth = info.extent.depth;
    
    res.mip_levels = info.mipLevels;
    res.array_layers = info.arrayLayers;
    
    res.format = info.format;
    
    res.aspect = resources.get_image(id)->get_default_aspect();
    return res;
  }
  
  gpu::ImageInfo RenderGraph::get_descriptor(ImageResourceId id) const {
    auto info = resources.get_info(id);
    gpu::ImageInfo res {};
    
    res.width = info.extent.width;
    res.height = info.extent.height;
    res.depth = info.extent.depth;
    
    res.mip_levels = info.mipLevels;
    res.array_layers = info.arrayLayers;
    
    res.format = info.format;
    
    res.aspect = resources.get_image(id)->get_default_aspect();
    return res;
  }

  RenderGraph::RenderGraph(gpu::Device &device, gpu::Swapchain &swapchain)
    : gpu {},
      resources {}
  {
    auto vk_backbuffers = gpu.take_backbuffers();
    backbuffers.reserve(vk_backbuffers.size());

    for (auto &img : vk_backbuffers) {
      backbuffers.push_back(resources.create_global_image_ref(img));
    }
    gpu.acquire_image();

    auto index = gpu.get_backbuf_index();
    if (index != 0) {
      resources.remap(backbuffers[0], backbuffers[index]);
    }
  }

  RenderGraph::~RenderGraph() {
    vkDeviceWaitIdle(gpu::app_device().api_device());
  }

  void RenderGraph::submit() {
    static int once = 2;
    tracking_state.flush(resources);
#if RENDERGRAPH_DEBUG
    tracking_state.dump_barriers();
#endif
    if (once) {
      tracking_state.dump_barriers();
      once--;
    }

    auto barriers = tracking_state.take_barriers();
    tracking_state.clear();

    gpu.begin();

    auto &api_cmd = gpu.get_cmdbuff(); 
    RenderResources res {resources, gpu};
    api_cmd.push_label("Rendergraph");
#if RENDERGRAPH_USE_EVENTS
    for (uint32_t i = 0; i < tasks.size(); i++) {
      api_cmd.push_label(tasks[i]->get_name().c_str());
      if (barriers.size() > i) {
        resolve_barrier(barriers, i, api_cmd.get_command_buffer());
      }

      tasks[i]->write_commands(res, api_cmd);
      api_cmd.end_renderpass(); //to be sure about barriers
      if (barriers.size() > i && barriers[i].signal_mask) {
        auto event = gpu.allocate_event();
        barriers[i].release_event = event;
        api_cmd.signal_event(event, barriers[i].signal_mask);
      }
      api_cmd.pop_label();
    }
#else
    for (uint32_t i = 0; i < tasks.size(); i++) {
      api_cmd.push_label(tasks[i]->get_name().c_str());
      if (barriers.size() > i) {
        write_barrier(barriers[i], api_cmd.get_command_buffer());
      }

      tasks[i]->write_commands(res, api_cmd);
      api_cmd.end_renderpass(); //to be sure about barriers
      api_cmd.pop_label();
    }
#endif
    tasks.clear();
    api_cmd.pop_label();
    
    if (!present_backbuffer) {
      gpu.submit(false);
      return;
    }

    if (gpu.get_backbuf_index() != 0) { //remap back to restore order
      resources.remap(backbuffers[0], backbuffers[gpu.get_backbuf_index()]);
    }

    gpu.submit(true);

    auto backbuffer_index = gpu.get_backbuf_index(); 
    if (backbuffer_index != 0) {
      resources.remap(backbuffers[0], backbuffers[backbuffer_index]);
    }
    present_backbuffer = false;
  }

  ImageResourceId RenderGraph::create_image(VkImageType type, const gpu::ImageInfo &info, VkImageTiling tiling, VkImageUsageFlags usage, gpu::ImageCreateOptions options) {
    return resources.create_global_image(ImageDescriptor {
      type,
      info.format,
      info.aspect,
      tiling,
      usage,
      info.width,
      info.height,
      info.depth,
      info.mip_levels,
      info.array_layers
    },
    options);
  }

  ImageResourceId RenderGraph::create_image(const ImageDescriptor &desc, gpu::ImageCreateOptions options) {
    return resources.create_global_image(desc, options);
  }

  BufferResourceId RenderGraph::create_buffer(VmaMemoryUsage mem, uint64_t size, VkBufferUsageFlags usage) {
    return resources.create_global_buffer(BufferDescriptor {size, usage, mem});
  }

  ImageResourceId RenderGraph::get_backbuffer() const {
    return backbuffers[0];
  }

  /*const gpu::ImageInfo &RenderGraph::get_descriptor(ImageResourceId id) const {
    return resources.get_info(id);
  }*/

  void RenderGraph::remap(ImageResourceId src, ImageResourceId dst) {
    resources.remap(src, dst);
  }

  void RenderGraph::write_barrier(const Barrier &barrier, VkCommandBuffer cmd) {
    if (barrier.is_empty()) {
      return;
    }
    
    std::vector<VkImageMemoryBarrier> image_barriers;
    std::vector<VkMemoryBarrier> mem_barriers;

    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags dst_stages = 0;

    for (const auto &state : barrier.image_barriers) {
      auto &image = resources.get_image(state.id.id);
      const auto &desc = resources.get_info(state.id.id);
      
      if (state.id.mip >= desc.mipLevels || state.id.layer >= desc.arrayLayers) {
        throw std::runtime_error {"Image subresource out of range"};
      }
      
      src_stages |= state.src.stages;
      dst_stages |= state.dst.stages;

      VkImageMemoryBarrier img_barrier {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        state.src.access,
        state.dst.access,
        state.src.layout,
        state.dst.layout,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image->api_image(),
        {image->get_full_aspect(), state.id.mip, 1, state.id.layer, 1}
      };
      image_barriers.push_back(img_barrier);
    }

    for (const auto &state : barrier.buffer_barriers) {
      src_stages |= state.src.stages;
      dst_stages |= state.dst.stages;

      VkMemoryBarrier mem_barrier {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        nullptr,
        state.src.access,
        state.dst.access
      };
      mem_barriers.push_back(mem_barrier);
    }

    if (!src_stages) {
      src_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    vkCmdPipelineBarrier(cmd, 
      src_stages, 
      dst_stages, 
      0, 
      mem_barriers.size(), 
      mem_barriers.data(), 
      0,
      nullptr,
      image_barriers.size(),
      image_barriers.data());
  }

  void RenderGraph::write_wait_events(const std::vector<Barrier> &barriers, const Barrier &barrier, VkCommandBuffer cmd) {
    if (barrier.is_empty()) {
      return;
    }
    
    std::vector<VkImageMemoryBarrier> image_barriers;
    std::vector<VkMemoryBarrier> mem_barriers;
    std::vector<VkEvent> wait_events;

    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags dst_stages = 0;

    for (const auto &state : barrier.image_barriers) {
      auto &image = resources.get_image(state.id.id);
      const auto &desc = resources.get_info(state.id.id);
      
      if (state.id.mip >= desc.mipLevels || state.id.layer >= desc.arrayLayers) {
        throw std::runtime_error {"Image subresource out of range"};
      }
      
      src_stages |= state.src.stages;
      dst_stages |= state.dst.stages;

      VkImageMemoryBarrier img_barrier {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        state.src.access,
        state.dst.access,
        state.src.layout,
        state.dst.layout,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image->api_image(),
        {image->get_full_aspect(), state.id.mip, 1, state.id.layer, 1}
      };
      
      image_barriers.push_back(img_barrier);

      if (state.wait_for == INVALID_BARRIER_INDEX) {
        throw std::runtime_error {"barrier waits for INVALID_BARRIER_INDEX"};
      }

      auto event = barriers.at(state.wait_for).release_event;

      if (!event) {
        throw std::runtime_error {"Use of not created event"};
      }

      wait_events.push_back(event);
    }

    for (const auto &state : barrier.buffer_barriers) {
      src_stages |= state.src.stages;
      dst_stages |= state.dst.stages;

      VkMemoryBarrier mem_barrier {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        nullptr,
        state.src.access,
        state.dst.access
      };
      mem_barriers.push_back(mem_barrier);

      if (state.wait_for == INVALID_BARRIER_INDEX) {
        throw std::runtime_error {"barrier waits for INVALID_BARRIER_INDEX"};
      }

      auto event = barriers.at(state.wait_for).release_event;

      if (!event) {
        throw std::runtime_error {"Use of not created event"};
      }

      wait_events.push_back(event);

    }

    if (!src_stages) {
      throw std::runtime_error {"Wait for nothing"};
    }

    vkCmdWaitEvents(cmd,
      wait_events.size(),
      wait_events.data(),
      src_stages,
      dst_stages,
      mem_barriers.size(),
      mem_barriers.data(),
      0,
      nullptr,
      image_barriers.size(),
      image_barriers.data());
  }

  void RenderGraph::resolve_barrier(const std::vector<Barrier> &barriers, uint32_t index, VkCommandBuffer cmd) {
    auto &barrier = barriers[index];
    if (barrier.is_empty()) {
      return;
    }

    std::vector<VkImageMemoryBarrier> image_barriers;
    std::vector<VkMemoryBarrier> mem_barriers;

    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags dst_stages = 0;

    for (const auto &state : barrier.image_barriers) {
      auto &image = resources.get_image(state.id.id);
      const auto &desc = resources.get_info(state.id.id);
      
      if (state.id.mip >= desc.mipLevels || state.id.layer >= desc.arrayLayers) {
        throw std::runtime_error {"Image subresource out of range"};
      }
      
      src_stages |= state.src.stages;
      dst_stages |= state.dst.stages;

      VkImageMemoryBarrier img_barrier {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        state.src.access,
        state.dst.access,
        state.src.layout,
        state.dst.layout,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image->api_image(),
        {image->get_full_aspect(), state.id.mip, 1, state.id.layer, 1}
      };
      
      image_barriers.push_back(img_barrier);
    }

    for (const auto &state : barrier.buffer_barriers) {
      src_stages |= state.src.stages;
      dst_stages |= state.dst.stages;

      VkMemoryBarrier mem_barrier {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        nullptr,
        state.src.access,
        state.dst.access
      };

      mem_barriers.push_back(mem_barrier);
    }

    if (!src_stages) {
      src_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    if (index == 0 || barrier.max_wait_task_index == index - 1) {
      vkCmdPipelineBarrier(cmd, 
        src_stages, 
        dst_stages, 
        0, 
        mem_barriers.size(), 
        mem_barriers.data(), 
        0,
        nullptr,
        image_barriers.size(),
        image_barriers.data());
      
      return;
    }

    std::vector<VkEvent> events;
    for (auto barrier_id : barrier.wait_tasks) {
      auto &src_barrier = barriers[barrier_id];

      if (!src_barrier.release_event) {
        throw std::runtime_error {"Event is not created!"};
      }
      
      src_stages |= src_barrier.signal_mask;
      events.push_back(src_barrier.release_event);
    }

  
    if (!events.size()) {
      throw std::runtime_error {"Events size == 0"};
    }

    vkCmdWaitEvents(cmd,
      events.size(),
      events.data(),
      src_stages,
      dst_stages,
      mem_barriers.size(),
      mem_barriers.data(),
      0,
      nullptr,
      image_barriers.size(),
      image_barriers.data());
  }

  gpu::BufferPtr &RenderResources::get_buffer(BufferResourceId id) {
    return resources.get_buffer(id);
  }
  
  gpu::ImagePtr &RenderResources::get_image(ImageResourceId id) {
    return resources.get_image(id);
  }
  
  VkImageView RenderResources::get_view(const ImageViewId &ref) {
    return get_image(ref.get_id())->get_view(ref.get_range());
  }
}