#ifndef DEPTH_AS_HPP_INCLUDED
#define DEPTH_AS_HPP_INCLUDED

#include "scene/camera.hpp"
#include "rendergraph/rendergraph.hpp"
#include "gpu/gpu.hpp"
#include "scene_renderer.hpp"

#include "image_readback.hpp"

struct TLASHolder {
  ~TLASHolder() { close(); } 
  void close();

  void create(gpu::TransferCmdPool &cmd_pool, const std::vector<VkAccelerationStructureKHR> &elems);
  void update(VkCommandBuffer cmd);

  VkAccelerationStructureKHR get_tlas() const {
    return tlas;
  }

private:
  void create_instance_buffer(const std::vector<VkAccelerationStructureKHR> &elems);

  VkAccelerationStructureKHR tlas {nullptr};
  gpu::BufferPtr tlas_storage_buffer;
  gpu::BufferPtr tlas_update_buffer;
  gpu::BufferPtr tlas_instance_buffer;

  uint32_t num_instances = 0;
};

struct DepthAs {

  ~DepthAs() { close(); }

  void close();
  void create(gpu::TransferCmdPool &cmd_pool, uint32_t width, uint32_t height);
  void update(VkCommandBuffer cmd, uint32_t num_primitives, const gpu::BufferPtr &src);

  VkAccelerationStructureKHR get_blas() const {
    return blas;
  }

  VkAccelerationStructureKHR get_tlas() const {
    return tlas_holder.get_tlas();
  }

private:
  void create_internal(uint32_t byte_size);

  VkAccelerationStructureKHR blas {nullptr};
  gpu::BufferPtr storage_buffer;
  gpu::BufferPtr update_buffer;

  TLASHolder tlas_holder;
};

struct DepthAsBuilder {
  ~DepthAsBuilder() {}

  void init(uint32_t width, uint32_t height);
  void run(rendergraph::RenderGraph &graph, DepthAs &depth_as, rendergraph::ImageResourceId depth, uint32_t mip, const DrawTAAParams &params);

private:
  gpu::ComputePipeline pipeline;
  gpu::BufferPtr aabb_storage;
};

struct UniqTriangleIDExtractor {
  UniqTriangleIDExtractor(rendergraph::RenderGraph &graph);

  void run(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId target);
  void process_readback(rendergraph::RenderGraph &graph, ReadBackSystem &readback_sys);

  rendergraph::BufferResourceId get_result() const { return reduce_buffer; }

private:
  rendergraph::BufferResourceId reduce_buffer;
  gpu::ComputePipeline reduce_pipeline;
  VkSampler integer_sampler {nullptr};

  ReadBackID readback_id = INVALID_READBACK;
  uint32_t readback_delay = 0;
  static constexpr uint32_t DELAY_FRAMES = 300;
};

struct TriangleAS {
  ~TriangleAS() { close(); }

  void close();
  void create(gpu::TransferCmdPool &ctx, uint32_t max_triangles);
  void update(VkCommandBuffer cmd, const gpu::BufferPtr &triangles_buffer, const gpu::BufferPtr &indirect_buffer);
  void update(VkCommandBuffer cmd, const gpu::BufferPtr &triangles_buffer, uint32_t triangles_count);

  VkAccelerationStructureKHR get_tlas() const {
    return tlas_holder.get_tlas();
  }

private:
  VkAccelerationStructureKHR blas {nullptr};
  gpu::BufferPtr blas_storage_buffer;
  gpu::BufferPtr blas_update_buffer;
  uint32_t max_triangles = 0;

  TLASHolder tlas_holder;
};

struct TriangleASBuilder {
  TriangleASBuilder(rendergraph::RenderGraph &graph, gpu::TransferCmdPool &ctx);
  void run(rendergraph::RenderGraph &graph, SceneRenderer &scene, rendergraph::ImageResourceId triangle_id_image, const glm::mat4 &camera);

  VkAccelerationStructureKHR get_tlas() const { return triangle_as.get_tlas(); }

  static constexpr uint32_t MAX_TRIANGLES = 1u << 17u;

private:
  UniqTriangleIDExtractor id_extractor;
  TriangleAS triangle_as;
  
  gpu::ComputePipeline indirect_pipeline;
  rendergraph::BufferResourceId indirect_compute; 

  gpu::ComputePipeline triangle_verts_pipeline;
  rendergraph::BufferResourceId triangle_verts;
  rendergraph::BufferResourceId as_indirect_args;
};

#endif