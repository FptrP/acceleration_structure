#include "depth_as.hpp"

#include <iostream>

void DepthAs::close() {
  auto device = gpu::app_device().api_device();
  
  if (blas)
    vkDestroyAccelerationStructureKHR(device, blas, nullptr);
  
  blas = nullptr;
  storage_buffer.release();
  update_buffer.release();

  if (tlas)
    vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
  
  tlas = nullptr;
  tlas_storage_buffer.release();
  tlas_update_buffer.release();
  tlas_instance_buffer.release();
}

static VkAccelerationStructureBuildSizesInfoKHR get_build_sizes(uint32_t width, uint32_t height) {
  auto device = gpu::app_device().api_device();
  VkAccelerationStructureGeometryKHR geometry {};
  geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
  geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

  VkAccelerationStructureBuildGeometryInfoKHR data_info {};
  data_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  data_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  data_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  data_info.geometryCount = 1;
  data_info.pGeometries = &geometry;
  //data_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t primitives = width * height;

  VkAccelerationStructureBuildSizesInfoKHR out {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &data_info, &primitives, &out);

  return out;
}

static gpu::BufferPtr fill_data(uint32_t width, uint32_t height) {
  auto storage_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(VkAabbPositionsKHR) * width * height,
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  float delta_x = 2.f/width;
  float delta_y = 2.f/height;

  auto *ptr = static_cast<VkAabbPositionsKHR*>(storage_buffer->get_mapped_ptr());

  for (uint32_t j = 0; j < height; j++) {
    
    float min_y = -1.f + j * delta_y; 
    float max_y = min_y + delta_y;

    for (uint32_t i = 0; i < width; i++) {
      float min_x = -1.f + i * delta_x;
      float max_x = min_x + delta_x;
      VkAabbPositionsKHR bbox {min_x, min_y, 0.1f, max_x, max_y, 1.f};
      ptr[j * width + i] = bbox;
    }
  }
  //workaround for TLAS AABB
  ptr[0].minZ -= 1000.f; //TOdo made normal TLAS update
  ptr[0].maxZ += 1000.f;
  return storage_buffer;
}

void DepthAs::create_internal(uint32_t byte_size) {
  auto device = gpu::app_device().api_device();
  storage_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, byte_size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

  VkAccelerationStructureCreateInfoKHR create_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
    .pNext = nullptr,
    .createFlags = 0,
    .buffer = storage_buffer->api_buffer(),
    .offset = 0,
    .size = byte_size,
    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    .deviceAddress = 0
  };

  vkCreateAccelerationStructureKHR(device, &create_info, nullptr, &blas);
}

static gpu::BufferPtr create_instance_buffer(VkAccelerationStructureKHR blas) {
  VkAccelerationStructureDeviceAddressInfoKHR address_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
    .pNext = nullptr,
    .accelerationStructure = blas
  };

  VkTransformMatrixKHR transform {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f
  };

  VkAccelerationStructureInstanceKHR instance {
    .transform = transform,
    .instanceCustomIndex = 0,
    .mask = 0xFF,
    .instanceShaderBindingTableRecordOffset = 0,
    .flags = 0,
    .accelerationStructureReference = vkGetAccelerationStructureDeviceAddressKHR(gpu::app_device().api_device(), &address_info)
  };

  auto result = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(instance),
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  auto *ptr = static_cast<decltype(instance)*>(result->get_mapped_ptr());
  *ptr = instance;
  return result;
}

void DepthAs::create_tlas_internal(gpu::TransferCmdPool &cmd_pool) {
  tlas_instance_buffer = create_instance_buffer(blas);

  VkAccelerationStructureGeometryInstancesDataKHR instances_data {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
    .pNext = nullptr,
    .arrayOfPointers = VK_FALSE,
    .data {.deviceAddress = tlas_instance_buffer->device_address()}
  };

  VkAccelerationStructureGeometryKHR geometry {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    .pNext = nullptr,
    .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
    .geometry {.instances = instances_data},
    .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
  };
  
  VkAccelerationStructureBuildGeometryInfoKHR data_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    .pNext = nullptr,
    .flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    .srcAccelerationStructure = nullptr,
    .dstAccelerationStructure = nullptr,
    .geometryCount = 1,
    .pGeometries = &geometry,
    .ppGeometries = nullptr,
    .scratchData {.hostAddress = nullptr}
  };

  uint32_t primitives = 1;

  VkAccelerationStructureBuildSizesInfoKHR out {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(gpu::app_device().api_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &data_info, &primitives, &out);
  std::cout << "AS = " << out.accelerationStructureSize << " BuildScrath " << out.buildScratchSize << " UpdateScratch " << out.updateScratchSize << "\n";

  tlas_storage_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, out.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
  
  if (out.updateScratchSize)
    tlas_update_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, out.updateScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  VkAccelerationStructureCreateInfoKHR create_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
    .pNext = nullptr,
    .createFlags = 0,
    .buffer = tlas_storage_buffer->api_buffer(),
    .offset = 0,
    .size = out.accelerationStructureSize,
    .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
    .deviceAddress = 0
  };

  VKCHECK(vkCreateAccelerationStructureKHR(gpu::app_device().api_device(), &create_info, nullptr, &tlas));

  auto scratch_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, out.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  data_info.dstAccelerationStructure = tlas;
  data_info.scratchData.deviceAddress = scratch_buffer->device_address();

  VkAccelerationStructureBuildRangeInfoKHR range {
    .primitiveCount = 1,
    .primitiveOffset = 0,
    .firstVertex = 0,
    .transformOffset = 0
  };

  auto range_ptr = &range;

  VkCommandBufferBeginInfo begin_info {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  auto cmd = cmd_pool.get_cmd_buffer();
  vkBeginCommandBuffer(cmd, &begin_info);
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &data_info, &range_ptr);
  vkEndCommandBuffer(cmd);
  cmd_pool.submit_and_wait();
}

void DepthAs::create(gpu::TransferCmdPool &cmd_pool, uint32_t width, uint32_t height) {
  close();

  auto sizes = get_build_sizes(width, height);

  std::cout << "AS = " << sizes.accelerationStructureSize << " BuildScrath " << sizes.buildScratchSize << " UpdateScratch " << sizes.updateScratchSize << "\n";
  
  create_internal(sizes.accelerationStructureSize);

  auto src_buffer = fill_data(width, height);
  if (sizes.updateScratchSize)
    update_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizes.updateScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  auto scratch_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizes.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  VkAccelerationStructureGeometryAabbsDataKHR aabbs {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
    .pNext = nullptr,
    .data = {.deviceAddress = src_buffer->device_address()},
    .stride = sizeof(VkAabbPositionsKHR)
  };

  VkAccelerationStructureGeometryKHR geometry {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    .pNext = nullptr,
    .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
    .geometry = {.aabbs = aabbs },
    .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
  };

  VkAccelerationStructureBuildGeometryInfoKHR mesh_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    .pNext = nullptr,
    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    .flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    .srcAccelerationStructure = nullptr,
    .dstAccelerationStructure = blas,
    .geometryCount = 1,
    .pGeometries = &geometry,
    .ppGeometries = nullptr,
    .scratchData {.deviceAddress = scratch_buffer->device_address()}
  };

  VkAccelerationStructureBuildRangeInfoKHR range {
    .primitiveCount = width * height,
    .primitiveOffset = 0,
    .firstVertex = 0,
    .transformOffset = 0
  };

  auto range_ptr = &range;

  VkCommandBufferBeginInfo begin_info {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  auto cmd = cmd_pool.get_cmd_buffer();
  vkBeginCommandBuffer(cmd, &begin_info);
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &mesh_info, &range_ptr);
  vkEndCommandBuffer(cmd);
  cmd_pool.submit_and_wait();

  create_tlas_internal(cmd_pool);
}

static void push_wr_barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier aa_memory_barrier {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr, 
    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT
  };

  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
    0, 1, &aa_memory_barrier, 0, nullptr, 0, nullptr);
}

static void push_rw_barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier aa_memory_barrier {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr, 
    .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
    .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT
  };

  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
    0, 1, &aa_memory_barrier, 0, nullptr, 0, nullptr);
}

void DepthAs::update(VkCommandBuffer cmd, uint32_t num_primitives, const gpu::BufferPtr &src) {
  VkAccelerationStructureGeometryAabbsDataKHR aabbs {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
    .pNext = nullptr,
    .data = {.deviceAddress = src->device_address()},
    .stride = sizeof(VkAabbPositionsKHR)
  };

  VkAccelerationStructureGeometryKHR geometry {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    .pNext = nullptr,
    .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
    .geometry = {.aabbs = aabbs },
    .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
  };

  VkAccelerationStructureBuildGeometryInfoKHR mesh_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    .pNext = nullptr,
    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    .flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR,
    .srcAccelerationStructure = blas,
    .dstAccelerationStructure = blas,
    .geometryCount = 1,
    .pGeometries = &geometry,
    .ppGeometries = nullptr,
    .scratchData {.deviceAddress = update_buffer->device_address()}
  };

  VkAccelerationStructureBuildRangeInfoKHR range {
    .primitiveCount = num_primitives,
    .primitiveOffset = 0,
    .firstVertex = 0,
    .transformOffset = 0
  };

  auto range_ptr = &range;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &mesh_info, &range_ptr);
  push_wr_barrier(cmd);
  update_tlas(cmd);
}

void DepthAs::update_tlas(VkCommandBuffer cmd) {
  VkAccelerationStructureGeometryInstancesDataKHR instances_data {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
    .pNext = nullptr,
    .arrayOfPointers = VK_FALSE,
    .data {.deviceAddress = tlas_instance_buffer->device_address()}
  };

  VkAccelerationStructureGeometryKHR geometry {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    .pNext = nullptr,
    .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
    .geometry {.instances = instances_data},
    .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
  };
  
  VkAccelerationStructureBuildGeometryInfoKHR data_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    .pNext = nullptr,
    .flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    .srcAccelerationStructure = tlas,
    .dstAccelerationStructure = tlas,
    .geometryCount = 1,
    .pGeometries = &geometry,
    .ppGeometries = nullptr,
    .scratchData {.deviceAddress = tlas_update_buffer->device_address()}
  };

  VkAccelerationStructureBuildRangeInfoKHR range {
    .primitiveCount = 1,
    .primitiveOffset = 0,
    .firstVertex = 0,
    .transformOffset = 0
  };

  auto range_ptr = &range;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &data_info, &range_ptr);
}

void DepthAs::test_update(gpu::TransferCmdPool &cmd_pool, uint32_t width, uint32_t height) {
  auto src_buffer = fill_data(width, height);
  auto *ptr = static_cast<VkAabbPositionsKHR*>(src_buffer->get_mapped_ptr());
  ptr[0] = VkAabbPositionsKHR {0.f, 0.f, -1.f, 2.f/width, 2.f/height, 1.f};

  VkCommandBufferBeginInfo begin_info {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  auto cmd = cmd_pool.get_cmd_buffer();
  vkBeginCommandBuffer(cmd, &begin_info);
  update(cmd, width * height, src_buffer);
  vkEndCommandBuffer(cmd);
  cmd_pool.submit_and_wait();
}

void DepthAsBuilder::init(uint32_t width, uint32_t height) {
  aabb_storage = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(VkAabbPositionsKHR) * width * height,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  
  pipeline = gpu::create_compute_pipeline("build_depth_as");
}

static VkExtent3D calculate_mip(VkExtent3D src, uint32_t mip) {
  for (uint32_t i = 0; i < mip; i++) {
    src.width = std::max((src.width + 1u)/2u, 1u);
    src.height = std::max((src.height + 1u)/2u, 1u);
    src.depth = std::max((src.depth + 1u)/2u, 1u);
  }
  return src;
} 

void DepthAsBuilder::run(rendergraph::RenderGraph &graph, DepthAs &depth_as, rendergraph::ImageResourceId depth, uint32_t mip, const DrawTAAParams &params) {
  struct Input {
    rendergraph::ImageViewId depth;
  };

  struct PushConstants {
    float fovy;
    float aspect;
    float min_z;
    float max_z;
  };

  PushConstants push_const {params.fovy_aspect_znear_zfar.x, params.fovy_aspect_znear_zfar.y, params.fovy_aspect_znear_zfar.z, params.fovy_aspect_znear_zfar.w};
  auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  graph.add_task<Input>("buildDepthBlas",
  [&](Input &input, rendergraph::RenderGraphBuilder &builder){
    input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, mip, 1, 0, 1);
  },
  [=, &depth_as](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
    auto api_cmd = cmd.get_command_buffer();

    push_rw_barrier(api_cmd);

    auto set = resources.allocate_set(pipeline.get_layout(0));
    gpu::write_set(set,
      gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
      gpu::SSBOBinding {1, aabb_storage});
    
    auto extent = resources.get_image(input.depth.get_id())->get_extent();
    extent = calculate_mip(extent, mip);

    cmd.bind_pipeline(pipeline);
    cmd.bind_descriptors_compute(0, {set});
    cmd.push_constants_compute(0, sizeof(push_const), &push_const);
    cmd.dispatch((extent.width + 7)/8, (extent.height + 3)/4, 1);


    push_wr_barrier(api_cmd);  
    
    depth_as.update(api_cmd, extent.width * extent.height, aabb_storage);

    push_wr_barrier(api_cmd);
  });
}