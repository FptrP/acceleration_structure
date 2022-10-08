#include "resources.hpp"

#include <iostream>

namespace rendergraph {
  
  static VkImageCreateFlags options_to_flags(gpu::ImageCreateOptions options) {
    switch (options) {
    case gpu::ImageCreateOptions::Array2D:
      return VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    case gpu::ImageCreateOptions::Cubemap:
      return VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    default:
      return 0;
    }
  }

  ImageResourceId GraphResources::create_global_image(const ImageDescriptor &desc, gpu::ImageCreateOptions options) {
    uint32_t image_index = global_images.size();
    
    uint32_t count = desc.array_layers * desc.mip_levels;
    std::unique_ptr<ImageTrackingState[]> ptr;
    ptr.reset(new ImageTrackingState[count]);

    global_images.emplace_back(GlobalImage {
      {}, 
      std::move(ptr)
    });

    VkImageCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = options_to_flags(options),
      .imageType = desc.type,
      .format = desc.format,
      .extent = VkExtent3D{desc.width, desc.height, desc.depth},
      .mipLevels = desc.mip_levels,
      .arrayLayers = desc.array_layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = desc.tiling,
      .usage = desc.usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    global_images.back().vk_image = gpu::create_driver_image(info); //create(desc.type, desc.get_vk_info(), desc.tiling, desc.usage, options);
    
    ImageResourceId id {};
    id.index = image_index;
    return id;
  }

  ImageResourceId GraphResources::create_global_image_ref(const gpu::ImagePtr &image) {
    uint32_t image_index = global_images.size();

    uint32_t count = image->get_array_layers() * image->get_mip_levels();
    std::unique_ptr<ImageTrackingState[]> ptr;
    ptr.reset(new ImageTrackingState[count]);

    global_images.emplace_back(GlobalImage {
      {}, 
      std::move(ptr)
    });

    global_images.back().vk_image = gpu::create_image_ref(image->api_image(), image->get_info());// create_reference(image.get_image(), desc);
    
    ImageResourceId id {};
    id.index = image_index;
    return id;
  }
  
  BufferResourceId GraphResources::create_global_buffer(const BufferDescriptor &desc) {
    uint32_t buffer_index = global_buffers.size();

    global_buffers.emplace_back(GlobalBuffer {
      {},
      {}
    });

    global_buffers.back().vk_buffer = gpu::create_buffer(desc.memory_type, desc.size, desc.usage);

    BufferResourceId id {};
    id.index = buffer_index;
    return id;
  }
  
  void GraphResources::remap(ImageResourceId src, ImageResourceId dst) {
    std::swap(global_images.at(src.index), global_images.at(dst.index));
  }
  
  void GraphResources::remap(BufferResourceId src, BufferResourceId dst) {
    //std::swap(buffer_remap.at(src.index), buffer_remap.at(dst.index));
    std::swap(global_buffers.at(src.index), global_buffers.at(dst.index));
  }

  const VkImageCreateInfo &GraphResources::get_info(ImageResourceId id) const {
    return global_images.at(id.index).vk_image->get_info(); 
  }

  gpu::ImagePtr &GraphResources::get_image(ImageResourceId id) {
    return global_images.at(id.index).vk_image;
  }

  const gpu::ImagePtr &GraphResources::get_image(ImageResourceId id) const {
    return global_images.at(id.index).vk_image;
  }
  
  gpu::BufferPtr &GraphResources::get_buffer(BufferResourceId id) {
    //auto index = buffer_remap.at(id.index);
    return global_buffers.at(id.index).vk_buffer;
  }

  const BufferTrackingState &GraphResources::get_resource_state(BufferResourceId id) const {
    //auto index = buffer_remap.at(id.index);
    return global_buffers.at(id.index).state;
  }
  
  const ImageTrackingState &GraphResources::get_resource_state(ImageSubresourceId id) const {
    auto &img = global_images.at(id.id.index); 
    auto mip_count = img.vk_image->get_mip_levels(); 
    return img.states[id.layer * mip_count + id.mip];
  }
    
  BufferTrackingState &GraphResources::get_resource_state(BufferResourceId id) {
    //auto index = buffer_remap.at(id.index);
    return global_buffers.at(id.index).state;
  }
  
  ImageTrackingState &GraphResources::get_resource_state(ImageSubresourceId id) {
    auto &img = global_images.at(id.id.index); 
    auto mip_count = img.vk_image->get_mip_levels(); 

    return img.states[id.layer * mip_count + id.mip];
  }

  static inline bool is_ro_access(VkAccessFlags flags) {
    const auto read_msk =
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|
      VK_ACCESS_INDEX_READ_BIT|
      VK_ACCESS_INDIRECT_COMMAND_READ_BIT|
      VK_ACCESS_INPUT_ATTACHMENT_READ_BIT|
      VK_ACCESS_MEMORY_READ_BIT|
      VK_ACCESS_SHADER_READ_BIT|
      VK_ACCESS_TRANSFER_READ_BIT|
      VK_ACCESS_UNIFORM_READ_BIT|
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;

    return (flags & read_msk);
  }

  static inline bool is_write_access(VkAccessFlags flags) {
    const auto write_msk = 
      VK_ACCESS_SHADER_WRITE_BIT|
      VK_ACCESS_TRANSFER_WRITE_BIT|
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|
      VK_ACCESS_MEMORY_WRITE_BIT;

    return (flags & write_msk);
  }

  static bool merge_states(ImageTrackingState &state, const ImageSubresourceState &access) {
    if (state.dst.layout != access.layout) {
      return false;
    }

    if (!is_write_access(state.dst.access) && !is_write_access(access.access)) {
      state.dst.stages |= access.stages;
      state.dst.access |= access.access;
      return true;
    }
    return false;
  }

  static void flush_barrier(std::vector<Barrier> &barriers, const ImageSubresourceId &id, const ImageTrackingState &track) {
    if (barriers.size() <= track.barrier_id) {
      barriers.resize(track.barrier_id + 1);
    }
    
    ImageBarrierState image_barrier {};
    image_barrier.id = id;
    image_barrier.wait_for = track.wait_for;
    image_barrier.src = track.src;
    image_barrier.dst = track.dst;

    barriers[track.barrier_id].image_barriers.push_back(image_barrier);
    if (image_barrier.wait_for != INVALID_BARRIER_INDEX) {
      barriers[image_barrier.wait_for].signal_mask |= image_barrier.src.stages;
    }
  }

  static void flush_barrier(std::vector<Barrier> &barriers, const BufferResourceId &id, const BufferTrackingState &track) {
    if (barriers.size() <= track.barrier_id) {
      barriers.resize(track.barrier_id + 1);
    }

    BufferBarrierState buffer_barrier {};
    buffer_barrier.id = id;
    buffer_barrier.wait_for = track.wait_for;
    buffer_barrier.src = track.src;
    buffer_barrier.dst = track.dst;

    barriers[track.barrier_id].buffer_barriers.push_back(buffer_barrier);
    if (buffer_barrier.wait_for != INVALID_BARRIER_INDEX) {
      barriers[buffer_barrier.wait_for].signal_mask |= buffer_barrier.src.stages;
    }
  }

  static void flush_resource(std::vector<TaskResources> &tasks, const ImageSubresourceId &id, const ImageTrackingState &track) {
    if (tasks.size() <= track.wait_for) {
      tasks.resize(track.wait_for + 1);
    }

    ImageReleaseState image_release {};
    image_release.acquire_at = track.barrier_id;
    image_release.id = id;
    image_release.src = track.src;
    image_release.dst = track.dst;

    tasks[track.wait_for].release_images.push_back(image_release);
    tasks[track.wait_for].release_index = std::min(tasks[track.wait_for].release_index, track.barrier_id);
    tasks[track.wait_for].stages |= track.src.stages;
  }

  static void flush_resource(std::vector<TaskResources> &tasks, const BufferResourceId &id, const BufferTrackingState &track) {
    if (tasks.size() <= track.wait_for) {
      tasks.resize(track.wait_for + 1);
    }

    BufferReleaseState buffer_release {};
    buffer_release.acquire_at = track.barrier_id;
    buffer_release.id = id;
    buffer_release.src = track.src;
    buffer_release.dst = track.dst;

    tasks[track.wait_for].release_buffers.push_back(buffer_release);
    tasks[track.wait_for].release_index = std::min(tasks[track.wait_for].release_index, track.barrier_id);
    tasks[track.wait_for].stages |= track.src.stages;
  }

  static bool is_empty_state(const BufferTrackingState &track) {
    if (track.barrier_id == INVALID_BARRIER_INDEX) {
      return true;
    }

    return false;
  }

  static bool is_empty_state(const ImageTrackingState &track) {
    if (track.barrier_id == INVALID_BARRIER_INDEX) {
      return true;
    }

    return false;
  }

  static void dump_stages(VkPipelineStageFlags flags);

  static bool validate_stage(VkPipelineStageFlags flags) {
    constexpr auto valid = 
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT|
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT|
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT|
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|
      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT|
      VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT|
      VK_PIPELINE_STAGE_TRANSFER_BIT|
      VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    
    return (flags & ~valid) == 0;
  }
  template <typename T>
  struct StateValidator {
    StateValidator(const T &t) : ref {t} {}
    ~StateValidator() {
      /*if (!validate_stage(ref.src.stages) || !validate_stage(ref.dst.stages)) {
        dump_stages(ref.src.stages);
        dump_stages(ref.dst.stages);
        std::cout << "Tracking error\n";
      }*/
    }

  private:
    const T &ref;
  };

  void TrackingState::add_input(GraphResources &resources, const BufferResourceId &id, const BufferState &state) {
    auto &track = resources.get_resource_state(id);
    StateValidator<decltype(track)> validator {track};

    if (is_empty_state(track)) { //acquire resource
      track.barrier_id = 0;
      track.last_access = index;
      track.wait_for = INVALID_BARRIER_INDEX;
      track.dst = state;
      dirty_buffers.push_back(id);
      return;
    }

    if (!is_write_access(track.dst.access) && !is_write_access(state.access)) {
      track.dst.access |= state.access;
      track.dst.stages |= state.stages;
      track.last_access = index;
      return;
    }
    //uncompatible accesses in the same task
    if (track.barrier_id == index) {
      throw std::runtime_error {"Incompatible buffer usage in task"};
    }

    if (track.wait_for != INVALID_BARRIER_INDEX) {
      flush_resource(task_resources, id, track);
    } else {
      flush_barrier(barriers, id, track);
    }

    track.barrier_id = index;
    track.wait_for = track.last_access;
    track.last_access = index;
    track.src = track.dst;
    track.dst = state;
  }
  
  void TrackingState::add_input(GraphResources &resources, const ImageSubresourceId &id, const ImageSubresourceState &state) {
    auto &track = resources.get_resource_state(id);
    StateValidator<decltype(track)> validator {track};

    if (is_empty_state(track)) { //acquire resource
      track.barrier_id = 0;
      track.last_access = index;
      track.wait_for = INVALID_BARRIER_INDEX;
      track.dst = state;
      dirty_images.push_back(id);
      return;
    }

    if (merge_states(track, state)) {
      track.last_access = index;
      return;
    }

    //uncompatible accesses in the same task
    if (track.barrier_id == index) {
      throw std::runtime_error {"Incompatible image usage in task"};
    }

    if (track.wait_for != INVALID_BARRIER_INDEX) {
      flush_resource(task_resources, id, track);
    } else {
      flush_barrier(barriers, id, track);
    }

    track.wait_for = track.last_access;
    track.last_access = index;
    track.barrier_id = index;
    track.src = track.dst;
    track.dst = state;
  }

  void TrackingState::flush(GraphResources &resources) {
    for (auto id : dirty_images) {
      auto &track = resources.get_resource_state(id);
      StateValidator<decltype(track)> validator {track};

      if (track.wait_for != INVALID_BARRIER_INDEX) {
        flush_resource(task_resources, id, track);
      } else {
        flush_barrier(barriers, id, track);
      }

      track.src = track.dst;
      track.barrier_id = INVALID_BARRIER_INDEX;
      track.last_access = INVALID_BARRIER_INDEX;
      track.wait_for = INVALID_BARRIER_INDEX;
    }

    for (auto id : dirty_buffers) {
      auto &track = resources.get_resource_state(id);
      StateValidator<decltype(track)> validator {track};

      if (track.wait_for != INVALID_BARRIER_INDEX) {
        flush_resource(task_resources, id, track);
      } else {
        flush_barrier(barriers, id, track);
      }

      track.src = track.dst;
      track.barrier_id = INVALID_BARRIER_INDEX;
      track.last_access = INVALID_BARRIER_INDEX;
      track.wait_for = INVALID_BARRIER_INDEX;
    }
    
    gen_barriers();
    gen_event_sync();

    index = 0;
    dirty_buffers.clear();
    dirty_images.clear();
  }

  void TrackingState::gen_barriers() {
    
    for (uint32_t index = 0; index < task_resources.size(); index++) {
      const auto &task = task_resources[index];
      uint32_t release_index = task.release_index;
      
      if (task.is_empty())
        continue;
      
      if (release_index == INVALID_BARRIER_INDEX) {
        throw std::runtime_error {"Invalid release index"};
      }

      if (barriers.size() <= release_index) {
        barriers.resize(release_index + 1);
      }

      auto &barrier = barriers[release_index];

      for (const auto &res : task.release_buffers) {
        BufferBarrierState buf_barrier {};
        buf_barrier.wait_for = index;
        buf_barrier.id = res.id;
        buf_barrier.src = res.src;
        buf_barrier.dst = res.dst;

        barrier.buffer_barriers.push_back(buf_barrier);
      }

      for (const auto &res : task.release_images) {
        ImageBarrierState img_barrier {};
        img_barrier.wait_for = index;
        img_barrier.id = res.id;
        img_barrier.src = res.src;
        img_barrier.dst = res.dst;
        barrier.image_barriers.push_back(img_barrier);
      }
      
      barrier.wait_tasks.emplace(index);

      if (barrier.max_wait_task_index == INVALID_BARRIER_INDEX) {
        barrier.max_wait_task_index = index;
      }
      barrier.max_wait_task_index = std::max(barrier.max_wait_task_index, index);
    }
  }

  void TrackingState::gen_event_sync() {
    for (uint32_t target_index = 1; target_index < barriers.size(); target_index++) {
      auto &barrier = barriers[target_index];
      if (barrier.max_wait_task_index == target_index - 1) {
        continue; //use pipeline_barrier
      }

      for (auto task_id : barrier.wait_tasks) {
        barriers[task_id].signal_mask = task_resources[task_id].stages;
      }
    }
  }

  void TrackingState::clear() {
    index = 0;
    dirty_buffers.clear();
    dirty_images.clear();
    barriers.clear();
    task_resources.clear();
  }


  #define PRINT_FLAG(flag_name) if (flags & flag_name) { \
    if (!first) std::cout << "|"; \
    std::cout << #flag_name ; \
    first = false; \
  } 

  static void dump_stages(VkPipelineStageFlags flags) {
    if (!flags) {
      std::cout << "0";
      return;
    }

    bool first = true;
    PRINT_FLAG(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_TRANSFER_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_HOST_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
    PRINT_FLAG(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
  }

  static void dump_access(VkAccessFlags flags) {
    if (!flags) {
      std::cout << "0";
      return;
    }

    bool first = true;
    PRINT_FLAG(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
    PRINT_FLAG(VK_ACCESS_INDEX_READ_BIT)
    PRINT_FLAG(VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT)
    PRINT_FLAG(VK_ACCESS_UNIFORM_READ_BIT)
    PRINT_FLAG(VK_ACCESS_INPUT_ATTACHMENT_READ_BIT)
    PRINT_FLAG(VK_ACCESS_SHADER_READ_BIT)
    PRINT_FLAG(VK_ACCESS_SHADER_WRITE_BIT)
    PRINT_FLAG(VK_ACCESS_COLOR_ATTACHMENT_READ_BIT)
    PRINT_FLAG(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
    PRINT_FLAG(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
    PRINT_FLAG(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
    PRINT_FLAG(VK_ACCESS_TRANSFER_READ_BIT)
    PRINT_FLAG(VK_ACCESS_TRANSFER_WRITE_BIT)
    PRINT_FLAG(VK_ACCESS_HOST_READ_BIT)
    PRINT_FLAG(VK_ACCESS_HOST_WRITE_BIT)
    PRINT_FLAG(VK_ACCESS_MEMORY_READ_BIT)
    PRINT_FLAG(VK_ACCESS_MEMORY_WRITE_BIT)
  }

  static void dump_layout(VkImageLayout layout) {
    #define PRINT(x) case x: std::cout << #x; break

    switch (layout) {
      PRINT(VK_IMAGE_LAYOUT_UNDEFINED);
      PRINT(VK_IMAGE_LAYOUT_GENERAL);
      PRINT(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_PREINITIALIZED);
      PRINT(VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL);
      PRINT(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
      default:
        std::cout << "EXT layout";
    }
  } 

  void TrackingState::dump_barrier(const Barrier &barrier) {

    //std::cout << "Task " << barrier_id << "\n";
    std::cout << "Max wait index " << barrier.max_wait_task_index << "\n";
    for (const auto &img_barrier : barrier.image_barriers) {
      std::cout << " - Image barrier " << "\n";
      std::cout << " --- id " << img_barrier.id.id.get_index() << "\n";
      std::cout << " --- mip = " << img_barrier.id.mip << " layer = " << img_barrier.id.layer << "\n";
      std::cout << " --- wait for " << img_barrier.wait_for << "\n";
      std::cout << " --- src_stages : "; dump_stages(img_barrier.src.stages); std::cout << "\n";
      std::cout << " --- src_access : "; dump_access(img_barrier.src.access); std::cout << "\n";
      std::cout << " --- src_layout : "; dump_layout(img_barrier.src.layout); std::cout << "\n";
      
      std::cout << " --- dst_stages : "; dump_stages(img_barrier.dst.stages); std::cout << "\n";
      std::cout << " --- dst_access : "; dump_access(img_barrier.dst.access); std::cout << "\n";
      std::cout << " --- dst_layout : "; dump_layout(img_barrier.dst.layout); std::cout << "\n";
    }

    for (const auto &buf_barrier : barrier.buffer_barriers) {

      std::cout << " - Memory barrier for buffer " << buf_barrier.id.get_index() << "\n";
      std::cout << " --- wait for " << buf_barrier.wait_for << "\n";
      std::cout << " --- src_stages : "; dump_stages(buf_barrier.src.stages); std::cout << "\n";
      std::cout << " --- src_access : "; dump_access(buf_barrier.src.access); std::cout << "\n";
      
      std::cout << " --- dst_stages : "; dump_stages(buf_barrier.dst.stages); std::cout << "\n";
      std::cout << " --- dst_access : "; dump_access(buf_barrier.dst.access); std::cout << "\n";
    }

    std::cout << "Signal = ";
    dump_stages(barrier.signal_mask);
    std::cout << "\n";
  }

  void TrackingState::dump_barriers() {
    for (uint32_t i = 0; i < barriers.size(); i++) {
      std::cout << "Barrier " << i << "\n";
      dump_barrier(barriers[i]);
    }
  } 

  void TrackingState::dump_task_resources(const TaskResources &res) {
    for (const auto &img_release : res.release_images) {
      std::cout << " - Image " << "\n";
      std::cout << " --- id " << img_release.id.id.get_index() << "\n";
      std::cout << " --- mip = " << img_release.id.mip << " layer = " << img_release.id.layer << "\n";
      std::cout << " --- acquired at " << img_release.acquire_at << "\n";
      std::cout << " --- src_stages : "; dump_stages(img_release.src.stages); std::cout << "\n";
      std::cout << " --- src_access : "; dump_access(img_release.src.access); std::cout << "\n";
      std::cout << " --- src_layout : "; dump_layout(img_release.src.layout); std::cout << "\n";
      
      std::cout << " --- dst_stages : "; dump_stages(img_release.dst.stages); std::cout << "\n";
      std::cout << " --- dst_access : "; dump_access(img_release.dst.access); std::cout << "\n";
      std::cout << " --- dst_layout : "; dump_layout(img_release.dst.layout); std::cout << "\n";
    }

    for (const auto &buf_release : res.release_buffers) {

      std::cout << " - Memory barrier for buffer " << buf_release.id.get_index() << "\n";
      std::cout << " --- acquire at " << buf_release.acquire_at << "\n";
      std::cout << " --- src_stages : "; dump_stages(buf_release.src.stages); std::cout << "\n";
      std::cout << " --- src_access : "; dump_access(buf_release.src.access); std::cout << "\n";
      
      std::cout << " --- dst_stages : "; dump_stages(buf_release.dst.stages); std::cout << "\n";
      std::cout << " --- dst_access : "; dump_access(buf_release.dst.access); std::cout << "\n";
    }
  }

  void TrackingState::dump_task_resources() {
    for (uint32_t i = 0; i < task_resources.size(); i++) {
      std::cout << "Task " << i << "\n";
      std::cout << "Stages "; dump_stages(task_resources[i].stages); std::cout << "\n";
      dump_task_resources(task_resources[i]);
    }
  }

}