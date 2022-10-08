#include "image_readback.hpp"

#include <cstring>

static uint32_t get_texel_size(VkFormat fmt) {
  switch (fmt)
  {
  case VK_FORMAT_R8_UNORM:
  case VK_FORMAT_R8_SNORM:
  case VK_FORMAT_R8_USCALED:
  case VK_FORMAT_R8_SSCALED:
  case VK_FORMAT_R8_UINT:
  case VK_FORMAT_R8_SINT:
  case VK_FORMAT_R8_SRGB:
    return 1;
  case VK_FORMAT_R8G8_UNORM:
  case VK_FORMAT_R8G8_SNORM:
  case VK_FORMAT_R8G8_USCALED:
  case VK_FORMAT_R8G8_SSCALED:
  case VK_FORMAT_R8G8_UINT:
  case VK_FORMAT_R8G8_SINT:
  case VK_FORMAT_R8G8_SRGB:
  case VK_FORMAT_R16_UNORM:
  case VK_FORMAT_R16_SNORM:
  case VK_FORMAT_R16_USCALED:
  case VK_FORMAT_R16_SSCALED:
  case VK_FORMAT_R16_UINT:
  case VK_FORMAT_R16_SINT:
  case VK_FORMAT_R16_SFLOAT:
  case VK_FORMAT_D16_UNORM:
  case VK_FORMAT_D16_UNORM_S8_UINT:
    return 2;
  case VK_FORMAT_R8G8B8A8_UNORM:
  case VK_FORMAT_R8G8B8A8_SNORM:
  case VK_FORMAT_R8G8B8A8_USCALED:
  case VK_FORMAT_R8G8B8A8_SSCALED:
  case VK_FORMAT_R8G8B8A8_UINT:
  case VK_FORMAT_R8G8B8A8_SINT:
  case VK_FORMAT_R8G8B8A8_SRGB:
  case VK_FORMAT_B8G8R8A8_UNORM:
  case VK_FORMAT_B8G8R8A8_SNORM:
  case VK_FORMAT_B8G8R8A8_USCALED:
  case VK_FORMAT_B8G8R8A8_SSCALED:
  case VK_FORMAT_B8G8R8A8_UINT:
  case VK_FORMAT_B8G8R8A8_SINT:
  case VK_FORMAT_B8G8R8A8_SRGB:
  case VK_FORMAT_R16G16_UNORM:
  case VK_FORMAT_R16G16_SNORM:
  case VK_FORMAT_R16G16_USCALED:
  case VK_FORMAT_R16G16_SSCALED:
  case VK_FORMAT_R16G16_UINT:
  case VK_FORMAT_R16G16_SINT:
  case VK_FORMAT_R16G16_SFLOAT:
  case VK_FORMAT_R32_UINT:
  case VK_FORMAT_R32_SINT:
  case VK_FORMAT_R32_SFLOAT:
  case VK_FORMAT_D32_SFLOAT:
  case VK_FORMAT_D24_UNORM_S8_UINT:
    return 4;
  default:
    throw std::runtime_error {"Unsupported format"};
    break;
  }
  return 0;
}

ReadBackID ReadBackSystem::read_image(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image) {
  return read_image(graph, image, 0, 0, 0);
}

ReadBackID ReadBackSystem::read_image(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, VkImageAspectFlags aspect, uint32_t mip, uint32_t layer) {
  struct TaskData {};

  const auto &desc = graph.get_descriptor(image);

  uint32_t image_width = std::max(1u, desc.width/(1u << mip));
  uint32_t image_height = std::max(1u, desc.height/(1u << mip));
  uint32_t texel_size = get_texel_size(desc.format); //stencil not supported
  VkImageAspectFlags image_aspect = (aspect != 0)? aspect : desc.aspect; 
  
  auto buf = gpu::create_buffer(VMA_MEMORY_USAGE_GPU_TO_CPU, image_height * image_width * texel_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto api_buffer = buf->api_buffer();
  
  ReadBackID id = next_request_id;
  next_request_id++;

  requests[id] = Request {graph.get_frames_count() + 1, image_width, image_height, desc.format, texel_size, std::move(buf)};

  graph.add_task<TaskData>("ImageRead",
    [&](TaskData &, rendergraph::RenderGraphBuilder &builder) {
      builder.transfer_read(image, mip, 1, layer, 1);
    },
    [=](TaskData &, rendergraph::RenderResources &resources, gpu::CmdContext &ctx) {
      auto api_cmd = ctx.get_command_buffer();
      auto api_image = resources.get_image(image)->api_image();
      
      VkBufferImageCopy region {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource {image_aspect, mip, layer, 1},
        .imageOffset {0, 0, 0},
        .imageExtent {image_width, image_height, 1}
      };

      vkCmdCopyImageToBuffer(api_cmd, api_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, api_buffer, 1, &region);
    });
  
  return id;
}

void ReadBackSystem::after_submit(rendergraph::RenderGraph &graph) {
  auto it = requests.begin();
  while(it != requests.end()) {
    if (it->second.wait_frames > 0) {
      it->second.wait_frames--;
      it++;
      continue;
    }

    auto id = it->first;
    auto &src_request = it->second;
    auto src_ptr = src_request.data->get_mapped_ptr();

    if (!src_ptr) {
      throw std::runtime_error {"Error, readback got unmapped cpu pointer"};
    }
    src_request.data->invalidate_mapped_memory();
    
    ReadBackData dst_request {};
    dst_request.width = src_request.width;
    dst_request.height = src_request.height;
    dst_request.texel_size = src_request.texel_size;
    dst_request.texel_fmt = src_request.texel_fmt;

    const uint32_t byte_count = dst_request.width * dst_request.height * dst_request.texel_size;
    dst_request.bytes.reset(new uint8_t[byte_count]);
    std::memcpy(dst_request.bytes.get(), src_ptr, byte_count);
    processed_requests[id] = std::move(dst_request);
  
    it = requests.erase(it);
  }

}

ReadBackData ReadBackSystem::get_data(ReadBackID id) {
  ReadBackData readback {};
  
  auto &src = processed_requests.at(id);
  std::swap(readback.bytes, src.bytes);
  std::swap(readback.width, src.width);
  std::swap(readback.height, src.height);
  std::swap(readback.texel_fmt, src.texel_fmt);
  std::swap(readback.texel_size, src.texel_size);
  processed_requests.erase(id);
  return readback;
}