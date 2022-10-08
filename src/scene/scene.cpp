#include "scene.hpp"


#include <iostream>
#include <unordered_map>
#include <glm/gtc/type_ptr.hpp>
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include <lib/tiny_gltf.h>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

namespace scene {

  gpu::VertexInput get_vertex_input() {
    gpu::VertexInput vinput;

    vinput.bindings = {{0, sizeof(scene::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
    vinput.attributes = {
      {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(scene::Vertex, pos)
      },
      {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(scene::Vertex, norm)
      },
      {
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(scene::Vertex, uv)
      }
    };
    
    return vinput;
  }

  gpu::VertexInput get_vertex_input_shadow() {
    gpu::VertexInput vinput;

    vinput.bindings = {{0, sizeof(scene::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
    vinput.attributes = {
      {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(scene::Vertex, pos)
      }
    };
    
    return vinput;
  }

  static void copy_data(gpu::TransferCmdPool &transfer_pool, gpu::BufferPtr &dst, gpu::BufferPtr &transfer, uint32_t byte_count, uint8_t *data) {
    
    VkCommandBufferBeginInfo begin_info {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = 0,
      .pInheritanceInfo = nullptr
    };
    
    uint32_t offset = 0;
    uint32_t remaining_size = byte_count;

    while (remaining_size) {
      auto chunk = std::min(remaining_size, (uint32_t)transfer->get_size());
      std::memcpy(transfer->get_mapped_ptr(), data, chunk);
      transfer->flush();
      
      VkBufferCopy region {
        .srcOffset = 0,
        .dstOffset = offset,
        .size = chunk
      };

      data += chunk;
      remaining_size -= chunk;
      offset += chunk;

      auto cmd = transfer_pool.get_cmd_buffer();

      vkBeginCommandBuffer(cmd, &begin_info);
      vkCmdCopyBuffer(cmd, transfer->api_buffer(), dst->api_buffer(), 1, &region);
      vkEndCommandBuffer(cmd);
      transfer_pool.submit_and_wait();
    }

  }

  struct MaterialDesc {
    std::string albedo_path;
    std::string mr_path;
  };

  static VkFilter gltf_remap_filter(int filter) {
    switch (filter) {
    case TINYGLTF_TEXTURE_FILTER_NEAREST:
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
      return VK_FILTER_NEAREST;
    case TINYGLTF_TEXTURE_FILTER_LINEAR:
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
      return VK_FILTER_LINEAR; 
    }
    return VK_FILTER_LINEAR;
  }

  static VkSamplerMipmapMode gltf_remap_mipmap_mode(int filter) {
    switch (filter)
    {
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR; 
    }
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }

  static VkSamplerAddressMode gltf_remap_address_mode(int mode) {
    switch (mode)
    {
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case TINYGLTF_TEXTURE_WRAP_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }

  static void tinygltf_load_materials(gpu::TransferCmdPool &transfer_pool, const fs::path folder, tinygltf::Model &model, CompiledScene &out_scene) {
    out_scene.images.reserve(model.images.size());
    for (uint32_t i = 0; i < model.images.size(); i++) {
      const auto &src = model.images[i];
      const std::string path = (folder / src.uri).native();
      out_scene.images.emplace_back(load_image_rgba8(transfer_pool, path.c_str()));
    }

    out_scene.samplers.reserve(model.samplers.size());
    for (auto &smp : model.samplers) {
      auto cfg = gpu::DEFAULT_SAMPLER;
      cfg.magFilter = gltf_remap_filter(smp.magFilter);
      cfg.minFilter = gltf_remap_filter(smp.minFilter);
      cfg.mipmapMode = gltf_remap_mipmap_mode(smp.minFilter);
      cfg.addressModeU = gltf_remap_address_mode(smp.wrapS);
      cfg.addressModeV = gltf_remap_address_mode(smp.wrapT);
      out_scene.samplers.push_back(gpu::create_sampler(cfg));
    }

    out_scene.textures.reserve(model.textures.size());
    for (auto &src : model.textures) {
      Texture tex;
      tex.image_index = src.source;
      tex.sampler_index = src.sampler;
      out_scene.textures.push_back(tex);
    }

    out_scene.materials.reserve(model.materials.size());
    for (uint32_t i = 0; i < model.materials.size(); i++) {
      const auto &src = model.materials[i];
      Material mat {};
      mat.albedo_tex_index = src.pbrMetallicRoughness.baseColorTexture.index;
      mat.metalic_roughness_index = src.pbrMetallicRoughness.metallicRoughnessTexture.index;
      mat.alpha_cutoff = src.alphaCutoff;
      mat.clip_alpha = src.alphaMode == "MASK";
      std::cout << "Material " << mat.albedo_tex_index << " " << mat.metalic_roughness_index << "\n";
      out_scene.materials.push_back(mat);
    }
  }

  static Primitive tinygltf_load_prim(const tinygltf::Model &model, const tinygltf::Primitive &src, std::vector<Vertex> &vertices, std::vector<uint32_t> &indexes) {
		const float *pos_ptr = nullptr;
		const float *norm_ptr = nullptr;
		const float *uv_ptr = nullptr;

    uint32_t first_index = static_cast<uint32_t>(indexes.size());
		uint32_t vertex_start = static_cast<uint32_t>(vertices.size());
		uint32_t index_count = 0;
    uint32_t vertex_count = 0;

    auto pos_it = src.attributes.find("POSITION");
    auto norm_it = src.attributes.find("NORMAL");
    auto tex_it = src.attributes.find("TEXCOORD_0");
    
    if (pos_it != src.attributes.end()) {
      auto &accessor = model.accessors[pos_it->second];
      auto &view = model.bufferViews[accessor.bufferView];
      pos_ptr = reinterpret_cast<const float*>(&model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
      vertex_count = accessor.count;
    } else {
      throw std::runtime_error {"No position"};
    }

    if (norm_it != src.attributes.end()) {
      auto &accessor = model.accessors[norm_it->second];
      auto &view = model.bufferViews[accessor.bufferView];
      norm_ptr = reinterpret_cast<const float*>(&model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
    }

    if (tex_it != src.attributes.end()) {
      auto &accessor = model.accessors[tex_it->second];
      auto &view = model.bufferViews[accessor.bufferView];
      uv_ptr = reinterpret_cast<const float*>(&model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
    }

    for (uint32_t i = 0; i < vertex_count; i++) {
      Vertex v {};

      if (pos_ptr) {
        v.pos = glm::vec3{pos_ptr[3 * i + 0], pos_ptr[3 * i + 1], pos_ptr[3 * i + 2]};
      }

      if (norm_ptr) {
        v.norm = glm::vec3{norm_ptr[3 * i + 0], norm_ptr[3 * i + 1], norm_ptr[3 * i + 2]};
      }

      if (uv_ptr) {
        v.uv = glm::vec2{uv_ptr[2 * i + 0], uv_ptr[2 * i + 1]};
      }
      vertices.push_back(v);
    }

    auto &accessor = model.accessors[src.indices];
    auto &view = model.bufferViews[accessor.bufferView];
    auto &buffer = model.buffers[view.buffer];

    index_count = accessor.count;
    if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT) {
      const uint32_t *buf = reinterpret_cast<const uint32_t*>(&buffer.data[accessor.byteOffset + view.byteOffset]);  
      for (uint32_t i = 0; i < index_count; i++) {
        indexes.push_back(buf[i]);
      }
    } else if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
      const uint16_t *buf = reinterpret_cast<const uint16_t*>(&buffer.data[accessor.byteOffset + view.byteOffset]);  
      for (uint32_t i = 0; i < index_count; i++) {
        indexes.push_back(buf[i]);
      }
    } else if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE) {
      const uint8_t *buf = reinterpret_cast<const uint8_t*>(&buffer.data[accessor.byteOffset + view.byteOffset]);  
      for (uint32_t i = 0; i < index_count; i++) {
        indexes.push_back(buf[i]);
      }
    } else {
      throw std::runtime_error {"Unsupported index type"};
    }

    Primitive prim {};
    prim.material_index = src.material;
    prim.index_count = index_count;
    prim.vertex_offset = vertex_start;
    prim.index_offset = first_index;

    std::cout << "Proccessed prim " << prim.vertex_offset << " " << prim.index_offset << " " << prim.index_count << "\n";
    return prim;
  } 

  static void tinygltf_load_meshes(gpu::TransferCmdPool &transfer_pool, const fs::path folder, const tinygltf::Model &model, CompiledScene &out_scene, bool for_ray_tracing) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indexes;

    out_scene.root_meshes.reserve(model.meshes.size());
    for (const auto &src : model.meshes) {
      BaseMesh base_mesh;
      base_mesh.primitives.reserve(src.primitives.size());

      for (const auto &prim : src.primitives) {
        auto res = tinygltf_load_prim(model, prim, vertices, indexes); 
        base_mesh.primitives.push_back(res);
      }

      out_scene.root_meshes.push_back(std::move(base_mesh));
    }
    //upload vertices
    const uint64_t verts_size = sizeof(Vertex) * vertices.size();
    const uint64_t index_size = sizeof(uint32_t) * indexes.size();

    VkBufferUsageFlags ray_tracing_flags = 0; 
    if (for_ray_tracing) {
      ray_tracing_flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }

    out_scene.vertex_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, verts_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|ray_tracing_flags);
    out_scene.index_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, index_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT|ray_tracing_flags);

    const uint32_t TRANSFER_SIZE = 10 * 1024;
    auto transfer_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, TRANSFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    
    copy_data(transfer_pool, out_scene.vertex_buffer, transfer_buffer, verts_size, (uint8_t*)vertices.data());
    copy_data(transfer_pool, out_scene.index_buffer, transfer_buffer, index_size, (uint8_t*)indexes.data());
  }

  static void tinygltf_load_nodes(const tinygltf::Model &model, const tinygltf::Node &src_node, BaseNode &out_node) {
    out_node.transform = glm::identity<glm::mat4>();
    out_node.mesh_index = src_node.mesh;

	  if (src_node.translation.size() == 3) {
		  out_node.transform = glm::translate(out_node.transform, glm::vec3(glm::make_vec3(src_node.translation.data())));
  	}
	  if (src_node.rotation.size() == 4) {
		  glm::quat q = glm::make_quat(src_node.rotation.data());
		  out_node.transform *= glm::mat4(q);
	  }
	  if (src_node.scale.size() == 3) {
		  out_node.transform = glm::scale(out_node.transform, glm::vec3(glm::make_vec3(src_node.scale.data())));
	  }

    if (src_node.matrix.size() == 16) {
      out_node.transform = glm::make_mat4x4(src_node.matrix.data());
    }
    out_node.children.resize(src_node.children.size());

    for (uint32_t i = 0; i < src_node.children.size(); i++) {
      tinygltf_load_nodes(model, model.nodes[src_node.children[i]], out_node.children[i]);
    }
  }

  CompiledScene load_tinygltf_scene(gpu::TransferCmdPool &transfer_pool, const std::string &path, bool for_ray_traing) {
    CompiledScene result_scene {};
    tinygltf::Model model;
    tinygltf::TinyGLTF loader {};
    std::string err, warn;
    
    if (!loader.LoadASCIIFromFile(&model, &err, &warn, path)) {
      throw std::runtime_error {err};
    }
    auto folder = fs::path{path}.parent_path();
    tinygltf_load_materials(transfer_pool, folder, model, result_scene);
    tinygltf_load_meshes(transfer_pool, folder, model, result_scene, for_ray_traing);
    
    int default_scene = std::max(0, model.defaultScene);
    const auto &scene = model.scenes[default_scene];

    result_scene.base_nodes.resize(scene.nodes.size());
    std::cout << "Loading scene " << default_scene << " with "<< scene.nodes.size() << " root nodes\n";  
    

    for (uint32_t i = 0; i < scene.nodes.size(); i++) {
      auto node_id = scene.nodes[i];
      std::cout << "Loading node " << node_id << "\n";
      tinygltf_load_nodes(model, model.nodes[node_id], result_scene.base_nodes[i]);
    }
    
    if (!warn.empty()) {
      std::cout << "[W] " << warn.c_str() << "\n";
    }
    return result_scene;
  }

}