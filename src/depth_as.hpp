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
  void update(VkCommandBuffer cmd, uint32_t num_primitives, const gpu::BufferPtr &src, bool rebuild = false);

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
  void checkerboard_init(rendergraph::RenderGraph &graph, DepthAs &depth_as, const DrawTAAParams &params);

private:
  uint32_t dst_width = 0;
  uint32_t dst_height = 0;
  
  gpu::ComputePipeline pipeline;
  gpu::ComputePipeline init_pipeline;
  gpu::BufferPtr aabb_storage;
  bool rebuild = true;
};

struct UniqTriangleIDExtractor {
  UniqTriangleIDExtractor(rendergraph::RenderGraph &graph);

  void run(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId target, SceneRenderer &scene, const glm::mat4 &view_projection);
  void process_readback(rendergraph::RenderGraph &graph, ReadBackSystem &readback_sys);

  rendergraph::BufferResourceId get_result() const { return reduce_buffer; }

private:
  uint32_t num_buckets = 128;

  rendergraph::BufferResourceId reduce_buffer;
  rendergraph::BufferResourceId triangles_per_bucket;
  rendergraph::BufferResourceId buckets;

  gpu::ComputePipeline reduce_pipeline;
  gpu::ComputePipeline bucket_reduce_pipeline;
  
  VkSampler integer_sampler {nullptr};

  gpu::ComputePipeline fill_buffer_pipeline;

  ReadBackID readback_id = INVALID_READBACK;
  uint32_t readback_delay = 0;
  static constexpr uint32_t DELAY_FRAMES = 300;
};

struct TriangleAS {
  ~TriangleAS() { close(); }

  void close();
  void create(gpu::TransferCmdPool &ctx, uint32_t max_triangles);
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
  void run(rendergraph::RenderGraph &graph, SceneRenderer &scene, rendergraph::ImageResourceId triangle_id_image, const glm::mat4 &camera, const glm::mat4 &projection);

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

struct GbufferCompressor {
  GbufferCompressor(rendergraph::RenderGraph &graph, gpu::TransferCmdPool &transfer_pool, uint32_t width, uint32_t height);

  void build_tree(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth, uint32_t depth_mip, rendergraph::ImageResourceId normal, const DrawTAAParams &params);
  
  VkAccelerationStructureKHR get_tlas() const { return depth_as.get_tlas(); } 

private:
  gpu::ComputePipeline clear_pass;
  gpu::ComputePipeline first_pass;
  gpu::ComputePipeline compress_mips;

  rendergraph::ImageResourceId tree_levels;
  rendergraph::BufferResourceId nodes;
  VkSampler sampler;

  rendergraph::BufferResourceId counter;
  rendergraph::BufferResourceId aabbs;
  rendergraph::BufferResourceId compressed_planes;

  uint32_t num_elems = 0;
  
  DepthAs depth_as;

  struct CompressedPlane {
    uint32_t packed_normal;
    uint32_t pos_x;
    uint32_t pos_y;
    uint32_t size_depth;
  };

  enum {
    CHECK_GAPS = 1,
    DO_NOT_UPDATE = 2
  };
  
  void process_level(rendergraph::RenderGraph &graph, const DrawTAAParams &params, uint32_t src_level, uint32_t flag);
};

#endif