#include "scene.hpp"

#include <string>
#include <memory>
#include <cmath>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include <lib/stb_image.h>

namespace scene {

  struct PixelsDeleter {
    void operator()(stbi_uc *ptr) const {
      STBI_FREE(ptr);
    }
  };

  static void copy_pixels(VkCommandBuffer cmd, gpu::ImagePtr &dst, gpu::BufferPtr &transfer);
  static void gen_image_mips(VkCommandBuffer cmd, gpu::ImagePtr &dst);

  gpu::ImagePtr load_image_rgba8(gpu::TransferCmdPool &transfer_pool, const char *path) {
    int x, y, comps;

    std::unique_ptr<stbi_uc, PixelsDeleter> pixels;
    pixels.reset(stbi_load(path, &x, &y, &comps, 4));
    
    if (!pixels) {
      throw std::runtime_error {stbi_failure_reason()};
    }
    
    uint32_t mips = std::floor(std::log2(std::max(x, y))) + 1;
    auto flags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
    //gpu::ImageInfo image_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, uint32_t(x), uint32_t(y), 1, mips, 1};
    //gpu::Image output_image {};
    //output_image.create(VK_IMAGE_TYPE_2D, image_info, VK_IMAGE_TILING_OPTIMAL, flags);

    auto output_image = gpu::create_tex2d(VK_FORMAT_R8G8B8A8_SRGB, uint32_t(x), uint32_t(y), mips, flags); 
    uint64_t buff_size = x * y * 4;
    auto transfer_buffer = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, buff_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(transfer_buffer->get_mapped_ptr(), pixels.get(), buff_size);
    pixels.release();

    auto cmd = transfer_pool.get_cmd_buffer();
    VkCommandBufferBeginInfo begin_info {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin_info);

    copy_pixels(cmd, output_image, transfer_buffer);
    gen_image_mips(cmd, output_image);

    vkEndCommandBuffer(cmd);
    transfer_pool.submit_and_wait();

    return output_image;
  }

  static void copy_pixels(VkCommandBuffer cmd, gpu::ImagePtr &dst, gpu::BufferPtr &transfer) {
    VkImageMemoryBarrier image_barrier {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = dst->api_image(),
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    
    const auto &desc = dst->get_info();

    VkBufferImageCopy copy_region {
      .bufferOffset = 0,
      .bufferRowLength = desc.extent.width,
      .bufferImageHeight = desc.extent.height,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {desc.extent.width, desc.extent.height, 1}
    };

    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 
      0, nullptr,
      0, nullptr,
      1, &image_barrier);

    vkCmdCopyBufferToImage(cmd, transfer->api_buffer(), dst->api_image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);
  }

  static void gen_image_mips(VkCommandBuffer cmd, gpu::ImagePtr &dst) {
    auto &desc = dst->get_info();
    for (uint32_t dst_mip = 1; dst_mip < desc.mipLevels; dst_mip++) {
      const uint32_t src_mip = dst_mip - 1;
      const int32_t src_width = desc.extent.width/(1 << src_mip);
      const int32_t src_height = desc.extent.height/(1 << src_mip);

      VkImageMemoryBarrier src_barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst->api_image(),
        .subresourceRange {VK_IMAGE_ASPECT_COLOR_BIT, src_mip, 1, 0, 1}
      };

      VkImageMemoryBarrier dst_barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst->api_image(),
        .subresourceRange {VK_IMAGE_ASPECT_COLOR_BIT, dst_mip, 1, 0, 1}
      };

      VkImageMemoryBarrier src_sampled_barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst->api_image(),
        .subresourceRange {VK_IMAGE_ASPECT_COLOR_BIT, src_mip, 1, 0, 1}
      };
      
      VkImageBlit blit_region {
        .srcSubresource {VK_IMAGE_ASPECT_COLOR_BIT, src_mip, 0, 1},
        .srcOffsets {{0, 0, 0}, {std::max((int)src_width, 1), std::max((int)src_height, 1), 1}},
        .dstSubresource {VK_IMAGE_ASPECT_COLOR_BIT, dst_mip, 0, 1},
        .dstOffsets {{0, 0, 0}, {std::max(src_width/2, 1), std::max(src_height/2, 1), 1}}
      };

      auto barriers = {src_barrier, dst_barrier};

      vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 
        0, nullptr,
        0, nullptr,
        2, barriers.begin());

      vkCmdBlitImage(cmd, 
        dst->api_image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst->api_image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit_region,
        VK_FILTER_LINEAR);

      vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &src_sampled_barrier);
    }

    VkImageMemoryBarrier src_sampled_barrier {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = dst->api_image(),
      .subresourceRange {VK_IMAGE_ASPECT_COLOR_BIT, dst->get_info().mipLevels - 1, 1, 0, 1}
    };

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &src_sampled_barrier);

  }

}