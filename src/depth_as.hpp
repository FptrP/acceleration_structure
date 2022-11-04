#ifndef DEPTH_AS_HPP_INCLUDED
#define DEPTH_AS_HPP_INCLUDED

#include "scene/camera.hpp"
#include "rendergraph/rendergraph.hpp"
#include "gpu/gpu.hpp"
#include "scene_renderer.hpp"

struct DepthAs {

  ~DepthAs() { close(); }

  void close();
  void create(gpu::TransferCmdPool &cmd_pool, uint32_t width, uint32_t height);
  void update(VkCommandBuffer cmd, uint32_t num_primitives, const gpu::BufferPtr &src);

  void test_update(gpu::TransferCmdPool &cmd_pool, uint32_t width, uint32_t height);

  VkAccelerationStructureKHR get_blas() const {
    return blas;
  }

  VkAccelerationStructureKHR get_tlas() const {
    return tlas;
  }

private:
  void create_internal(uint32_t byte_size);
  void create_tlas_internal(gpu::TransferCmdPool &cmd_pool);
  void update_tlas(VkCommandBuffer cmd);

  VkAccelerationStructureKHR blas {nullptr};
  gpu::BufferPtr storage_buffer;
  gpu::BufferPtr update_buffer;

  VkAccelerationStructureKHR tlas {nullptr};
  gpu::BufferPtr tlas_storage_buffer;
  gpu::BufferPtr tlas_update_buffer;
  gpu::BufferPtr tlas_instance_buffer;
};

struct DepthAsBuilder {
  ~DepthAsBuilder() {}

  void init(uint32_t width, uint32_t height);
  void run(rendergraph::RenderGraph &graph, DepthAs &depth_as, rendergraph::ImageResourceId depth, uint32_t mip, const DrawTAAParams &params);

private:
  gpu::ComputePipeline pipeline;
  gpu::BufferPtr aabb_storage;
};

//void build_depth_as(rendergraph::RenderGraph &graph, DepthAs &depth_as, rendergraph::ImageResourceId depth, uint32_t mip = 0);

#endif