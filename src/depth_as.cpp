#include "depth_as.hpp"
#include "util_passes.hpp"
#include <iostream>

void TLASHolder::close() {
  auto device = gpu::app_device().api_device();
  if (tlas)
    vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
  
  num_instances = 0;
  tlas = nullptr;
  tlas_storage_buffer.release();
  tlas_instance_buffer.release();
  tlas_update_buffer.release();
}

void TLASHolder::create_instance_buffer(const std::vector<VkAccelerationStructureKHR> &elems) {
  VkAccelerationStructureDeviceAddressInfoKHR address_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
    .pNext = nullptr,
    .accelerationStructure = nullptr
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
    .accelerationStructureReference = 0
  };

  tlas_instance_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(instance) * elems.size(),
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  auto *ptr = static_cast<decltype(instance)*>(tlas_instance_buffer->get_mapped_ptr());
  
  for (uint32_t index = 0; index < elems.size(); index++) {
    address_info.accelerationStructure = elems[index];
    auto address = vkGetAccelerationStructureDeviceAddressKHR(gpu::app_device().api_device(), &address_info);
    instance.instanceCustomIndex = index;
    instance.accelerationStructureReference = address;
    ptr[index] = instance;
  }
}

void TLASHolder::create(gpu::TransferCmdPool &cmd_pool, const std::vector<VkAccelerationStructureKHR> &elems) {
  num_instances = elems.size();
  create_instance_buffer(elems);

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

  uint32_t primitives = num_instances;

  VkAccelerationStructureBuildSizesInfoKHR out {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(gpu::app_device().api_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &data_info, &primitives, &out);
  std::cout << "AS = " << out.accelerationStructureSize << " BuildScrath " << out.buildScratchSize << " UpdateScratch " << out.updateScratchSize << "\n";

  tlas_storage_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, out.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
  
  if (out.updateScratchSize)
    tlas_update_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, out.updateScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 128);

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
    .primitiveCount = primitives,
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

void TLASHolder::update(VkCommandBuffer cmd) {
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
    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR,
    .srcAccelerationStructure = tlas,
    .dstAccelerationStructure = tlas,
    .geometryCount = 1,
    .pGeometries = &geometry,
    .ppGeometries = nullptr,
    .scratchData {.deviceAddress = tlas_update_buffer->device_address()}
  };

  VkAccelerationStructureBuildRangeInfoKHR range {
    .primitiveCount = num_instances,
    .primitiveOffset = 0,
    .firstVertex = 0,
    .transformOffset = 0
  };

  auto range_ptr = &range;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &data_info, &range_ptr);
}

void DepthAs::close() {
  tlas_holder.close();

  auto device = gpu::app_device().api_device();
  
  if (blas)
    vkDestroyAccelerationStructureKHR(device, blas, nullptr);
  
  blas = nullptr;
  storage_buffer.release();
  update_buffer.release();
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

  tlas_holder.create(cmd_pool, {blas});
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
  tlas_holder.update(cmd);
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

UniqTriangleIDExtractor::UniqTriangleIDExtractor(rendergraph::RenderGraph &graph) {

  const uint32_t CANDIDATES_COUNT = (1 << 17);
  constexpr VkBufferUsageFlags hash_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  reduce_buffer = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, (1 + CANDIDATES_COUNT) * sizeof(uint32_t), hash_flags);
  
  reduce_pipeline = gpu::create_compute_pipeline("image_id_reduce");
  bucket_reduce_pipeline = gpu::create_compute_pipeline("buckets_reduce");

  auto desc = gpu::DEFAULT_SAMPLER;
  desc.minFilter = VK_FILTER_NEAREST;
  desc.magFilter = VK_FILTER_NEAREST;
  integer_sampler = gpu::create_sampler(desc);

  fill_buffer_pipeline = gpu::create_compute_pipeline("fill_triangles");

  num_buckets = 128;
  const uint32_t IDS_IN_BUCKET = 1024;
  triangles_per_bucket = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(uint32_t) * num_buckets, hash_flags);
  buckets = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(uint32_t) * num_buckets * IDS_IN_BUCKET, hash_flags);
}

void UniqTriangleIDExtractor::run(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId target, SceneRenderer &scene, const glm::mat4 &view_projection) {
  buffer_clear(graph, triangles_per_bucket, 0);
  buffer_clear(graph, buckets, 0);

  struct Nil {};

  graph.add_task<Nil>("ClearTriangles", 
  [&](Nil &input, rendergraph::RenderGraphBuilder &builder){
    builder.use_storage_buffer(reduce_buffer, VK_SHADER_STAGE_COMPUTE_BIT, false);
  },
  [=](Nil &input, rendergraph::RenderResources &res, gpu::CmdContext  &ctx) {
    auto set = res.allocate_set(fill_buffer_pipeline, 0);
    
    gpu::write_set(set, 
      gpu::SSBOBinding {0, res.get_buffer(reduce_buffer)});

    const uint32_t COUNT = 80000;

    ctx.bind_pipeline(fill_buffer_pipeline);
    ctx.bind_descriptors_compute(0, {set}, {});
    ctx.push_constants_compute(0, sizeof(COUNT), &COUNT);
    ctx.dispatch((COUNT + 31)/32, 1, 1);
  });

  struct Data {
    rendergraph::ImageViewId id_image;
  };
  
  graph.add_task<Data>("IdReduce", 
  [&](Data &input, rendergraph::RenderGraphBuilder &builder){
    input.id_image = builder.sample_image(target, VK_SHADER_STAGE_COMPUTE_BIT);
    builder.use_storage_buffer(triangles_per_bucket, VK_SHADER_STAGE_COMPUTE_BIT, false);
    builder.use_storage_buffer(buckets, VK_SHADER_STAGE_COMPUTE_BIT, false);
  },
  [=](Data &input, rendergraph::RenderResources &res, gpu::CmdContext  &ctx) {
    auto desc = res.get_image(input.id_image)->get_extent();

    auto set = res.allocate_set(reduce_pipeline, 0);

    gpu::write_set(set, 
      gpu::TextureBinding {0, res.get_view(input.id_image), integer_sampler},
      gpu::SSBOBinding {1, res.get_buffer(buckets)},
      gpu::SSBOBinding {2, res.get_buffer(triangles_per_bucket)});

    ctx.bind_pipeline(reduce_pipeline);
    ctx.bind_descriptors_compute(0, {set}, {});
    ctx.push_constants_compute(0, sizeof(num_buckets), &num_buckets);
    ctx.dispatch((desc.width + 31)/32, (desc.height + 31)/32, 1);
  });

  graph.add_task<Nil>("IdUnique", 
  [&](Nil &input, rendergraph::RenderGraphBuilder &builder){
    builder.use_storage_buffer(triangles_per_bucket, VK_SHADER_STAGE_COMPUTE_BIT, true);
    builder.use_storage_buffer(buckets, VK_SHADER_STAGE_COMPUTE_BIT, true);
    builder.use_storage_buffer(reduce_buffer, VK_SHADER_STAGE_COMPUTE_BIT, false);
  },
  [=](Nil &input, rendergraph::RenderResources &res, gpu::CmdContext  &ctx) {
    auto set = res.allocate_set(bucket_reduce_pipeline, 0);

    gpu::write_set(set, 
      gpu::SSBOBinding {0, res.get_buffer(triangles_per_bucket)},
      gpu::SSBOBinding {1, res.get_buffer(buckets)},
      gpu::SSBOBinding {2, res.get_buffer(reduce_buffer)});

    ctx.bind_pipeline(bucket_reduce_pipeline);
    ctx.bind_descriptors_compute(0, {set}, {});
    ctx.dispatch(num_buckets, 1, 1);
  });
}

void UniqTriangleIDExtractor::process_readback(rendergraph::RenderGraph &graph, ReadBackSystem &readback_sys) {
  if (readback_id == INVALID_READBACK) {
    if (readback_delay > 0) {
      readback_delay--;
      return;
    }
    
    readback_id = readback_sys.read_buffer(graph, reduce_buffer, 0, sizeof(uint32_t));
    readback_delay = DELAY_FRAMES;
    return;
  }

  if (!readback_sys.is_data_available(readback_id))
    return;
  auto result = readback_sys.get_data(readback_id);
  auto ptr = (const uint32_t*)result.bytes.get();
  std::cout << "TriangleID after reduce " << *ptr << "\n";
  readback_id = INVALID_READBACK;
}

void TriangleAS::close() {
  tlas_holder.close();
  
  auto device = gpu::app_device().api_device();
  if (blas)
    vkDestroyAccelerationStructureKHR(device, blas, nullptr);

  max_triangles = 0;
  blas = nullptr;
  blas_storage_buffer.release();
  blas_update_buffer.release();
}

struct Triangle {
  glm::vec3 v0;
  glm::vec3 v1;
  glm::vec3 v2;
};

gpu::BufferPtr make_triangle_buffer(uint32_t max_triangles_count) {
  Triangle triangle {};
  triangle.v0 = glm::vec3{-1.f, -1.f, 0.f};
  triangle.v1 = glm::vec3{ 1.f, -1.f, 0.f};
  triangle.v2 = glm::vec3{ 0.f,  1.f, 0.f};

  //triangle.v0 = glm::vec3{ 0.f,  0.f, 0.f};
  //triangle.v1 = glm::vec3{ 0.f,  0.f, 0.f};
  //triangle.v2 = glm::vec3{ 0.f,  0.f, 0.f};

  auto storage_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(Triangle) * max_triangles_count,
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  
  auto triangles = (Triangle*)storage_buffer->get_mapped_ptr();
  for (uint32_t i = 0; i < max_triangles_count; i++) {
    triangle.v0.z = i/5.f;
    triangle.v1.z = i/5.f;
    triangle.v2.z = i/5.f;
    triangles[i] = triangle;
  }

  return storage_buffer;
}

void TriangleAS::create(gpu::TransferCmdPool &ctx, uint32_t max_triangles_count) {
  close();

  max_triangles = max_triangles_count;
  auto temp_triangle_buffer = make_triangle_buffer(max_triangles);

  VkAccelerationStructureGeometryTrianglesDataKHR triangles_data {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
    .pNext = nullptr,
    .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
    .vertexData {.deviceAddress = temp_triangle_buffer->device_address() },
    .vertexStride = sizeof(glm::vec3),
    .maxVertex = 3 * max_triangles - 1,
    .indexType = VK_INDEX_TYPE_NONE_KHR,
    .indexData {.hostAddress = nullptr},
    .transformData {.hostAddress = nullptr} 
  };

  VkAccelerationStructureGeometryKHR geometry {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    .pNext = nullptr,
    .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
    .geometry = {.triangles = triangles_data },
    .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
  };

  VkAccelerationStructureBuildGeometryInfoKHR mesh_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    .pNext = nullptr,
    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    .flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    .srcAccelerationStructure = nullptr,
    .dstAccelerationStructure = nullptr,
    .geometryCount = 1,
    .pGeometries = &geometry,
    .ppGeometries = nullptr,
    .scratchData {.hostAddress = nullptr}
  };

  uint32_t primitives = max_triangles;

  VkAccelerationStructureBuildSizesInfoKHR out {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(gpu::app_device().api_device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &mesh_info, &primitives, &out);

  std::cout << "TriangleAS ";
  std::cout << "size = " << out.accelerationStructureSize << " ";
  std::cout << "buildSize = " << out.buildScratchSize << " ";
  std::cout << "updateSize = " << out.updateScratchSize << "\n";

  blas_storage_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, out.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

  if (out.updateScratchSize)
    blas_update_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, out.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 128);

  auto blas_build_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, out.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 128);

  VkAccelerationStructureCreateInfoKHR create_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
    .pNext = nullptr,
    .createFlags = 0,
    .buffer = blas_storage_buffer->api_buffer(),
    .offset = 0,
    .size = out.accelerationStructureSize,
    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    .deviceAddress = 0
  };

  VKCHECK(vkCreateAccelerationStructureKHR(gpu::app_device().api_device(), &create_info, nullptr, &blas));

  mesh_info.dstAccelerationStructure = blas;
  mesh_info.scratchData.deviceAddress = blas_build_buffer->device_address();

  VkAccelerationStructureBuildRangeInfoKHR range {
    .primitiveCount = max_triangles,
    .primitiveOffset = 0,
    .firstVertex = 0,
    .transformOffset = 0
  };

  auto range_ptr = &range;

  VkCommandBufferBeginInfo begin_info {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  auto cmd = ctx.get_cmd_buffer();
  vkBeginCommandBuffer(cmd, &begin_info);
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &mesh_info, &range_ptr);
  vkEndCommandBuffer(cmd);
  ctx.submit_and_wait();

  tlas_holder.create(ctx, {blas});
}

void TriangleAS::update(VkCommandBuffer cmd, const gpu::BufferPtr &triangles_buffer, uint32_t triangles_count) {
  VkAccelerationStructureGeometryTrianglesDataKHR triangles_data {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
    .pNext = nullptr,
    .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
    .vertexData {.deviceAddress = triangles_buffer->device_address() },
    .vertexStride = sizeof(glm::vec3),
    .maxVertex = 3 * triangles_count - 1,
    .indexType = VK_INDEX_TYPE_NONE_KHR,
    .indexData {.hostAddress = nullptr},
    .transformData {.hostAddress = nullptr} 
  };

  VkAccelerationStructureGeometryKHR geometry {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    .pNext = nullptr,
    .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
    .geometry = {.triangles = triangles_data },
    .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
  };

  VkAccelerationStructureBuildGeometryInfoKHR mesh_info {
    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    .pNext = nullptr,
    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    .flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    .srcAccelerationStructure = blas,
    .dstAccelerationStructure = blas,
    .geometryCount = 1,
    .pGeometries = &geometry,
    .ppGeometries = nullptr,
    .scratchData {.deviceAddress = blas_update_buffer->device_address()}
  };

  //auto range_address = indirect_buffer->device_address();
  //uint32_t stride = sizeof(VkAccelerationStructureBuildRangeInfoKHR);
  //const uint32_t *primitive_ptr = &max_triangles;
  //vkCmdBuildAccelerationStructuresIndirectKHR(cmd, 1, &mesh_info, &range_address, &stride, &primitive_ptr);
  
  VkAccelerationStructureBuildRangeInfoKHR range {0, 0, 0, 0};
  range.primitiveCount = triangles_count;
  auto range_ptr = &range;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &mesh_info, &range_ptr);
  push_wr_barrier(cmd);

  tlas_holder.update(cmd);
}

TriangleASBuilder::TriangleASBuilder(rendergraph::RenderGraph &graph, gpu::TransferCmdPool &ctx)
  : id_extractor {graph}
{
  auto max_triangles = 1u << 17u;
  triangle_as.create(ctx, max_triangles);

  VkBufferUsageFlags indirect_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  VkBufferUsageFlags as_flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  indirect_pipeline = gpu::create_compute_pipeline("unique_uints_init");
  indirect_compute = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(VkDispatchIndirectCommand), indirect_flags);

  triangle_verts_pipeline = gpu::create_compute_pipeline("create_triangles");
  triangle_verts = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(Triangle) * max_triangles, as_flags|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  as_indirect_args = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(VkAccelerationStructureBuildRangeInfoKHR), indirect_flags);
}

void TriangleASBuilder::run(rendergraph::RenderGraph &graph, SceneRenderer &scene, rendergraph::ImageResourceId triangle_id_image, const glm::mat4 &camera, const glm::mat4 &projection) {
  buffer_clear(graph, triangle_verts, 0);
  id_extractor.run(graph, triangle_id_image, scene, projection * camera);

  auto reduce_buffer = id_extractor.get_result();

  struct Nil {};
  graph.add_task<Nil>("FillIndirect",
  [&](Nil &, rendergraph::RenderGraphBuilder &builder){
    builder.use_storage_buffer(reduce_buffer, VK_SHADER_STAGE_COMPUTE_BIT, true);
    builder.use_storage_buffer(indirect_compute, VK_SHADER_STAGE_COMPUTE_BIT, false);
  },
  [=](Nil &, rendergraph::RenderResources &res, gpu::CmdContext &cmd){
    auto set = res.allocate_set(indirect_pipeline, 0);
    gpu::write_set(set,
      gpu::SSBOBinding {0, res.get_buffer(indirect_compute)},
      gpu::SSBOBinding {1, res.get_buffer(reduce_buffer)});

    cmd.bind_pipeline(indirect_pipeline);
    cmd.bind_descriptors_compute(0, {set}, {});
    cmd.dispatch(1, 1, 1);
  });

  auto verts_buffer = scene.get_target().vertex_buffer;
  auto index_buffer = scene.get_target().index_buffer;
  auto primitive_buffer = scene.get_target().primitive_buffer;
  auto transform_buffer = scene.get_scene_transforms();

  graph.add_task<Nil>("FillTriangles",
  [&](Nil &, rendergraph::RenderGraphBuilder &builder){
    builder.use_indirect_buffer(indirect_compute);

    builder.use_storage_buffer(reduce_buffer, VK_SHADER_STAGE_COMPUTE_BIT, true);
    builder.use_storage_buffer(transform_buffer, VK_SHADER_STAGE_COMPUTE_BIT, true);

    builder.use_storage_buffer(triangle_verts, VK_SHADER_STAGE_COMPUTE_BIT, false);
    builder.use_storage_buffer(as_indirect_args, VK_SHADER_STAGE_COMPUTE_BIT, false);
  },
  [=](Nil &, rendergraph::RenderResources &res, gpu::CmdContext &cmd){
    auto set = res.allocate_set(triangle_verts_pipeline, 0);
    gpu::write_set(set,
      gpu::SSBOBinding {0, res.get_buffer(transform_buffer)},
      gpu::SSBOBinding {1, verts_buffer},
      gpu::SSBOBinding {2, index_buffer},
      gpu::SSBOBinding {3, primitive_buffer},
      gpu::SSBOBinding {4, res.get_buffer(reduce_buffer)},
      gpu::SSBOBinding {5, res.get_buffer(as_indirect_args)},
      gpu::SSBOBinding {6, res.get_buffer(triangle_verts)});

    cmd.bind_pipeline(triangle_verts_pipeline);
    cmd.bind_descriptors_compute(0, {set}, {});
    cmd.push_constants_compute(0, sizeof(camera), &camera);
    cmd.dispatch_indirect(res.get_buffer(indirect_compute)->api_buffer());
  });

  graph.add_task<Nil>("UpdateAccelerationStructure",
  [&](Nil &, rendergraph::RenderGraphBuilder &builder){
    builder.use_indirect_buffer(as_indirect_args);
  },
  [=](Nil &, rendergraph::RenderResources &res, gpu::CmdContext &cmd){
    push_wr_barrier(cmd.get_command_buffer());
    //triangle_as.update(cmd.get_command_buffer(), res.get_buffer(triangle_verts), res.get_buffer(as_indirect_args));
    triangle_as.update(cmd.get_command_buffer(), res.get_buffer(triangle_verts), 80000);
    push_wr_barrier(cmd.get_command_buffer());
  });

}

GbufferCompressor::GbufferCompressor(rendergraph::RenderGraph &graph, gpu::TransferCmdPool &transfer_pool, uint32_t width, uint32_t height)
{
  num_elems = width * height;

  uint32_t mip_levels = std::floor(std::log2(std::max(width, height))) + 1u;

  gpu::ImageInfo info {VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width, height, 1, mip_levels, 1};
  tree_levels = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  clear_pass = gpu::create_compute_pipeline("tree_clear");
  first_pass = gpu::create_compute_pipeline("tree_init");
  compress_mips = gpu::create_compute_pipeline("tree_process");
  
  VkBufferUsageFlags as_flags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                              | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  counter = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  aabbs = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(VkAabbPositionsKHR) * num_elems, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|as_flags);
  compressed_planes = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(CompressedPlane) * num_elems, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  depth_as.create(transfer_pool, width, height);
}

void GbufferCompressor::build_tree(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth, uint32_t depth_mip, rendergraph::ImageResourceId normal, const DrawTAAParams &params)
{
  struct Nil {};
  graph.add_task<Nil>("ClearAABB", 
  [&](Nil &, rendergraph::RenderGraphBuilder &builder){
    builder.use_storage_buffer(aabbs, VK_SHADER_STAGE_COMPUTE_BIT, false);
  },
  [=](Nil &, rendergraph::RenderResources &res, gpu::CmdContext &cmd){
    auto set = res.allocate_set(clear_pass, 0);
    gpu::write_set(set, gpu::SSBOBinding {0, res.get_buffer(aabbs)});

    cmd.bind_pipeline(clear_pass);
    cmd.bind_descriptors_compute(0, {set});
    cmd.push_constants_compute(0, sizeof(num_elems), &num_elems);
    cmd.dispatch((num_elems + 31u)/32u, 1, 1);
  });

  clear_color(graph, tree_levels, VkClearColorValue {.float32 {0.f, 0.f, -1.f, 0}});
  buffer_clear(graph, counter, 0u);

  glm::mat4 normal_mat = glm::transpose(glm::inverse(params.camera));

  struct InitStruct {
    rendergraph::ImageViewId depth, normal, first_level;
  };
  
  graph.add_task<InitStruct>("InitTreeBuilding", 
  [&](InitStruct &input, rendergraph::RenderGraphBuilder &builder){
    input.normal = builder.sample_image(normal, VK_SHADER_STAGE_COMPUTE_BIT);
    input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_mip, 1, 0, 1);
    input.first_level = builder.use_storage_image(tree_levels, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
  },
  [=](InitStruct &input, rendergraph::RenderResources &res, gpu::CmdContext &cmd){
    auto set = res.allocate_set(first_pass, 0);
    gpu::write_set(set, 
      gpu::TextureBinding {0, res.get_view(input.depth), sampler},
      gpu::TextureBinding {1, res.get_view(input.normal), sampler},
      gpu::StorageTextureBinding {2, res.get_view(input.first_level)});
    
    auto extent = res.get_image(input.first_level)->get_extent();
    cmd.bind_pipeline(first_pass);
    cmd.bind_descriptors_compute(0, {set});
    cmd.push_constants_compute(0, sizeof(normal_mat), &normal_mat);
    cmd.dispatch((extent.width + 7)/8, (extent.height + 3)/4, 1);
  });

  uint32_t mips_count = graph.get_descriptor(tree_levels).mip_levels;
  if (mips_count < 2)
    throw std::runtime_error {"2 mips or more required"};
  
  uint32_t last_src_mip = std::min(5u, mips_count - 2); 
  for (uint32_t i = 0; i <= last_src_mip; i++) {
    process_level(graph, params, i, (i == 0)? CHECK_GAPS : (i == last_src_mip)? DO_NOT_UPDATE : 0u);
  }

  graph.add_task<Nil>("BuildAABB_AS", 
  [&](Nil &, rendergraph::RenderGraphBuilder &builder){
    builder.use_storage_buffer(aabbs, VK_SHADER_STAGE_COMPUTE_BIT, true);
  },
  [=](Nil &, rendergraph::RenderResources &res, gpu::CmdContext &cmd){
    push_rw_barrier(cmd.get_command_buffer());
    depth_as.update(cmd.get_command_buffer(), num_elems, res.get_buffer(aabbs));
    push_wr_barrier(cmd.get_command_buffer());
  });
  
}

void GbufferCompressor::process_level(rendergraph::RenderGraph &graph, const DrawTAAParams &params, uint32_t src_level, uint32_t flag) {
  struct Input {
    rendergraph::ImageViewId src;
    rendergraph::ImageViewId dst;
  };

  struct PushConstants {
    float aspect;
    float fovy;
    float znear;
    float zfar;
    uint32_t flag;
    uint32_t src_level;
  };

  PushConstants push_consts {params.fovy_aspect_znear_zfar.y,
                             params.fovy_aspect_znear_zfar.x,
                             params.fovy_aspect_znear_zfar.z,
                             params.fovy_aspect_znear_zfar.w,
                             flag, src_level};

  graph.add_task<Input>("ProcessLevel",
  [&](Input &input, rendergraph::RenderGraphBuilder &builder){
    input.src = builder.use_storage_image(tree_levels, VK_SHADER_STAGE_COMPUTE_BIT, src_level, 0);
    input.dst = builder.use_storage_image(tree_levels, VK_SHADER_STAGE_COMPUTE_BIT, src_level + 1, 0);
    
    builder.use_storage_buffer(counter, VK_SHADER_STAGE_COMPUTE_BIT, false);
    builder.use_storage_buffer(aabbs, VK_SHADER_STAGE_COMPUTE_BIT, false);
    builder.use_storage_buffer(compressed_planes, VK_SHADER_STAGE_COMPUTE_BIT, false);
  },
  [=](Input &input, rendergraph::RenderResources &res, gpu::CmdContext &cmd){
    auto set = res.allocate_set(compress_mips, 0);
    gpu::write_set(set, 
      gpu::StorageTextureBinding {0, res.get_view(input.src)},
      gpu::StorageTextureBinding {1, res.get_view(input.dst)},
      gpu::SSBOBinding {2, res.get_buffer(counter)},
      gpu::SSBOBinding {3, res.get_buffer(aabbs)},
      gpu::SSBOBinding {4, res.get_buffer(compressed_planes)});
    
    auto extent = res.get_image(input.src)->get_extent();
    for (uint32_t i = 0; i < src_level + 1; i++) {
      extent.width = std::max((extent.width + 1u)/2u, 1u);
      extent.height = std::max((extent.height + 1u)/2u, 1u);
    }

    cmd.bind_pipeline(compress_mips);
    cmd.bind_descriptors_compute(0, {set});
    cmd.push_constants_compute(0, sizeof(push_consts), &push_consts);
    cmd.dispatch((extent.width + 7)/8, (extent.height + 3)/4, 1);
  });
}