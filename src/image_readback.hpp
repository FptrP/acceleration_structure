#ifndef IMAGE_READBACK_HPP_INCLUDED
#define IMAGE_READBACK_HPP_INCLUDED

#include <memory>
#include <vector>
#include <unordered_map>

#include "rendergraph/rendergraph.hpp"

struct ReadBackData {
  uint32_t width = 0;
  uint32_t height = 0;
  VkFormat texel_fmt = VK_FORMAT_UNDEFINED;
  uint32_t texel_size = 0;
  std::unique_ptr<uint8_t[]> bytes {nullptr};
};

using ReadBackID = uint64_t;
const ReadBackID INVALID_READBACK = ~0ull; 

struct ReadBackSystem {
  ReadBackID read_image(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image);
  ReadBackID read_image(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, VkImageAspectFlags aspect, uint32_t mip, uint32_t layer);

  void after_submit(rendergraph::RenderGraph &graph);
  bool is_data_available(ReadBackID id) const { return processed_requests.count(id); }

  ReadBackData get_data(ReadBackID id);
  void clear();
private:

  struct Request {
    uint32_t wait_frames;
    uint32_t width;
    uint32_t height;
    VkFormat texel_fmt;
    uint32_t texel_size;
    gpu::BufferPtr data;
  };

  ReadBackID next_request_id = 0;
  std::unordered_map<ReadBackID, Request> requests;
  std::unordered_map<ReadBackID, ReadBackData> processed_requests;
};

#endif