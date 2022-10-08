#ifndef SCENE_HPP_INCLUDED
#define SCENE_HPP_INCLUDED

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <lib/volk.h>
#include "gpu/gpu.hpp"

namespace scene {
  constexpr uint32_t INVALID_TEXTURE = UINT32_MAX;

  struct Vertex {
    glm::vec3 pos;
    glm::vec3 norm;
    glm::vec2 uv;
  };

  struct Primitive {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t index_count;
    uint32_t material_index;
  };

  struct BaseMesh {
    std::vector<Primitive> primitives;
  };

  struct BaseNode {
    glm::mat4 transform;
    std::vector<BaseNode> children;
    int mesh_index;
  };

  struct Mesh {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t index_count;
    uint32_t material_index;
  };

  struct Material {
    uint32_t albedo_tex_index = INVALID_TEXTURE;
    uint32_t metalic_roughness_index = INVALID_TEXTURE;
    bool clip_alpha = false;
    float alpha_cutoff = 0.f;
  };

  struct Node {
    glm::mat4 transform;
    std::vector<uint32_t> meshes;
    std::vector<std::unique_ptr<Node>> children;
  };

  struct Texture {
    uint32_t image_index;
    uint32_t sampler_index;
  };

  struct CompiledScene {
    CompiledScene() {}

    CompiledScene(CompiledScene &&) = default;
    CompiledScene &operator=(CompiledScene &&) = default;

    CompiledScene(CompiledScene &) = delete;
    CompiledScene &operator=(CompiledScene &) = delete;

    std::vector<Material> materials;
    gpu::BufferPtr vertex_buffer;
    gpu::BufferPtr index_buffer;
    std::vector<gpu::ImagePtr> images;
    
    std::vector<VkSampler> samplers;
    std::vector<Texture> textures;
    std::vector<BaseMesh> root_meshes;
    std::vector<BaseNode> base_nodes;
  };

  gpu::VertexInput get_vertex_input();
  gpu::VertexInput get_vertex_input_shadow();

  CompiledScene load_tinygltf_scene(gpu::TransferCmdPool &transfer_pool, const std::string &path, bool for_ray_traing = true);
  gpu::ImagePtr load_image_rgba8(gpu::TransferCmdPool &transfer_pool, const char *path);
}

#endif