#include "scene_as.hpp"
#include <iostream>

namespace scene {

  SceneAccelerationStructure::~SceneAccelerationStructure() {
    auto vk_device = gpu::app_device().api_device(); 
    
    if (tlas)
      vkDestroyAccelerationStructureKHR(vk_device, tlas, nullptr);
    
    for (auto blas : blas_array) {
      if (blas) {
        vkDestroyAccelerationStructureKHR(vk_device, blas, nullptr);
      }
    }
  }

  void SceneAccelerationStructure::build(gpu::TransferCmdPool &transfer_pool, const CompiledScene &source) {
    for (const auto &mesh : source.root_meshes) { 
      build_blas(transfer_pool, mesh, source);
    }
    build_tlas(transfer_pool, source);
  }

  void SceneAccelerationStructure::build_blas(gpu::TransferCmdPool &transfer_pool, const BaseMesh &mesh, const CompiledScene &source) {
    uint32_t verts_count = source.vertex_buffer->get_size()/sizeof(Vertex);
    if (!verts_count) {
      verts_count = 1;
    }
  
    std::vector<VkAccelerationStructureGeometryKHR> geometry_data;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> prim_data;
    std::vector<uint32_t> geometry_prims;

    geometry_data.reserve(mesh.primitives.size());
    prim_data.reserve(mesh.primitives.size());
    geometry_prims.reserve(mesh.primitives.size());

    VkAccelerationStructureGeometryTrianglesDataKHR triangles {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
      .pNext = nullptr,
      .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
      .vertexData = VkDeviceOrHostAddressConstKHR {.deviceAddress = source.vertex_buffer->device_address()},
      .vertexStride = sizeof(Vertex),
      .maxVertex = verts_count - 1,
      .indexType = VK_INDEX_TYPE_UINT32,
      .indexData = VkDeviceOrHostAddressConstKHR {.deviceAddress = source.index_buffer->device_address()},
      .transformData = VkDeviceOrHostAddressConstKHR {.hostAddress = nullptr}
    };
    
    VkAccelerationStructureGeometryKHR geometry {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .pNext = nullptr,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .geometry {.triangles = triangles},
      .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    for (const auto &prim : mesh.primitives) {
      VkAccelerationStructureBuildRangeInfoKHR build_range {
        .primitiveCount = prim.index_count/3,
        .primitiveOffset = uint32_t(prim.index_offset * sizeof(uint32_t)),
        .firstVertex = prim.vertex_offset,
        .transformOffset = 0
      };

      geometry_data.push_back(geometry);
      geometry_prims.push_back(prim.index_count/3);
      prim_data.push_back(build_range);
    }

    
    VkAccelerationStructureBuildGeometryInfoKHR mesh_info {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .pNext = nullptr,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .srcAccelerationStructure = nullptr,
      .dstAccelerationStructure = nullptr,
      .geometryCount = (uint32_t)geometry_data.size(),
      .pGeometries = geometry_data.data(),
      .ppGeometries = nullptr,
      .scratchData = VkDeviceOrHostAddressKHR {.hostAddress = nullptr}
    };

    auto vk_device = gpu::app_device().api_device(); 

    VkAccelerationStructureBuildSizesInfoKHR build_info {};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;


    vkGetAccelerationStructureBuildSizesKHR(vk_device, 
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, 
      &mesh_info,
      geometry_prims.data(),
      &build_info);
    
    std::cout << build_info.accelerationStructureSize/1024 << "\n";
  
    auto storage_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, build_info.accelerationStructureSize,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    VkAccelerationStructureCreateInfoKHR create_info {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .pNext = nullptr,
      .createFlags = 0,
      .buffer = storage_buffer->api_buffer(),
      .offset = 0,
      .size = build_info.accelerationStructureSize,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .deviceAddress = 0
    };

    VkAccelerationStructureKHR acceleration_struct = nullptr;
    VKCHECK(vkCreateAccelerationStructureKHR(vk_device, &create_info, nullptr, &acceleration_struct));

    blas_array.push_back(acceleration_struct);
    blas_buffers.push_back(std::move(storage_buffer));

    auto scratch_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, build_info.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    auto range_ptr = prim_data.data();
    
    mesh_info.dstAccelerationStructure = acceleration_struct;
    mesh_info.scratchData.deviceAddress = scratch_buffer->device_address();

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    auto cmd = transfer_pool.get_cmd_buffer();
    vkBeginCommandBuffer(cmd, &begin_info);
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &mesh_info, &range_ptr);
    vkEndCommandBuffer(cmd);
    transfer_pool.submit_and_wait();
  }
  
  struct TLASNode {
    glm::mat4 transform;
    uint32_t acceleration_struct;
  };

  static void flattern_nodes(std::vector<TLASNode> &out, const BaseNode &node, const glm::mat4 &pre_transform) {
    TLASNode out_node;
    out_node.transform = pre_transform * node.transform;
    auto mesh_id = node.mesh_index;

    if (mesh_id >= 0) {
      out_node.acceleration_struct = mesh_id;
      out.push_back(out_node);
    }

    for (const auto &child : node.children) {
      flattern_nodes(out, child, out_node.transform);
    }
  }

  void SceneAccelerationStructure::build_tlas(gpu::TransferCmdPool &transfer_pool, const CompiledScene &source) {
    //todo : normal alghorithm
    VkTransformMatrixKHR transform {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f
    };

    VkAccelerationStructureInstanceKHR instance{};
		instance.transform = transform;
		instance.instanceCustomIndex = 0;
		instance.mask = 0xFF;
		instance.instanceShaderBindingTableRecordOffset = 0;
		instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		instance.accelerationStructureReference = 0;

    std::vector<TLASNode> nodes;
    for (auto &node : source.base_nodes) {
      glm::mat4 initial_transform = glm::identity<glm::mat4>();
      flattern_nodes(nodes, node, initial_transform);
    }
    

  
    auto instance_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(instance) * nodes.size(),
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

    auto instance_data = (VkAccelerationStructureInstanceKHR*)instance_buffer->get_mapped_ptr();

    auto vk_device = gpu::app_device().api_device();

    for (uint32_t i = 0; i < nodes.size(); i++) {
      const auto &tlas_node = nodes[i];

      VkAccelerationStructureDeviceAddressInfoKHR address_info {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .pNext = nullptr,
        .accelerationStructure = blas_array[tlas_node.acceleration_struct]
      };

      for (uint32_t y = 0; y < 3; y++) {
        for (uint32_t x = 0; x < 4; x++) {
          instance.transform.matrix[y][x] = tlas_node.transform[x][y];
        }
      }

      instance.accelerationStructureReference = vkGetAccelerationStructureDeviceAddressKHR(vk_device, &address_info);
      instance_data[i] = instance;
    }

    VkAccelerationStructureGeometryKHR geometry {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry.geometry.instances.arrayOfPointers = VK_FALSE;
		geometry.geometry.instances.data = VkDeviceOrHostAddressConstKHR {.deviceAddress = instance_buffer->device_address()};

    VkAccelerationStructureBuildGeometryInfoKHR build_geometry {};
		build_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_geometry.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		build_geometry.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		build_geometry.geometryCount = 1;
		build_geometry.pGeometries = &geometry;

    uint32_t primitive_count = source.root_meshes.size();
    VkAccelerationStructureBuildSizesInfoKHR build_sizes {};
    build_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  
    vkGetAccelerationStructureBuildSizesKHR(
			vk_device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&build_geometry,
			&primitive_count,
			&build_sizes);
    
    std::cout << "TLAS : " << build_sizes.accelerationStructureSize << "\n";
  
    tlas_memory = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, build_sizes.accelerationStructureSize,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    VkAccelerationStructureCreateInfoKHR create_info {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .pNext = nullptr,
      .createFlags = 0,
      .buffer = tlas_memory->api_buffer(),
      .offset = 0,
      .size = build_sizes.accelerationStructureSize,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .deviceAddress = 0
    };

    VKCHECK(vkCreateAccelerationStructureKHR(vk_device, &create_info, nullptr, &tlas));

    auto scratch_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, build_sizes.buildScratchSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    VkAccelerationStructureBuildRangeInfoKHR build_range {
      .primitiveCount = primitive_count,
      .primitiveOffset = 0,
      .firstVertex = 0,
      .transformOffset = 0
    };

    auto range_ptr = &build_range;
    build_geometry.dstAccelerationStructure = tlas;
    build_geometry.scratchData.deviceAddress = scratch_buffer->device_address();

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    auto cmd = transfer_pool.get_cmd_buffer();
    vkBeginCommandBuffer(cmd, &begin_info);
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_geometry, &range_ptr);
    vkEndCommandBuffer(cmd);
    transfer_pool.submit_and_wait();
  }

}