#ifndef SCENE_AS_HPP_INCLUDED
#define SCENE_AS_HPP_INCLUDED

#include "scene.hpp"

namespace scene {

  struct SceneAccelerationStructure {
    ~SceneAccelerationStructure();

    void build(gpu::TransferCmdPool &transfer_pool, const CompiledScene &source);
    void build_blas(gpu::TransferCmdPool &transfer_pool, const BaseMesh &mesh, const CompiledScene &source);
    void build_tlas(gpu::TransferCmdPool &transfer_pool, const CompiledScene &source);

    std::vector<gpu::BufferPtr> blas_buffers;
    std::vector<VkAccelerationStructureKHR> blas_array;

    gpu::BufferPtr tlas_memory;
    VkAccelerationStructureKHR tlas {nullptr};
  };



}

#endif