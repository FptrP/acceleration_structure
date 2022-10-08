#include "gpu_transfer.hpp"
#include "gpu/gpu.hpp"

#include <unordered_set>
#include <cstring>

namespace gpu_transfer {

  struct IdHash {
    std::size_t operator()(const rendergraph::BufferResourceId &id) const {
      return id.get_index();
    }
  };
  
  struct TransferBlock {
    rendergraph::BufferResourceId dst;
    uint64_t dst_offset;
    uint64_t src_offset;
    uint64_t size;
  };
  
  struct TransferState {
    void try_upload(rendergraph::BufferResourceId id, uint64_t offset, uint64_t size, const void *data) {
      if (write_offset + size > MAX_TRANSFER_SIZE) {
        throw std::runtime_error {"Uploading error"};
      }

      auto ptr = static_cast<uint8_t*>(transfer_buffers[buffer_id]->get_mapped_ptr());
      std::memcpy(ptr + write_offset, data, size);

      TransferBlock block {id, offset, write_offset, size};
      blocks.push_back(block);
      write_offset += size;
      dirty_buffers.emplace(id);
    }

    std::unordered_set<rendergraph::BufferResourceId, IdHash> dirty_buffers; 
    std::vector<TransferBlock> blocks;

    uint64_t write_offset = 0;
    uint32_t buffer_id = 0;
    std::vector<gpu::BufferPtr> transfer_buffers;
  };

  TransferState *g_transfer_state = nullptr;

  void init(const rendergraph::RenderGraph &graph) {
    close();

    g_transfer_state = new TransferState {};

    auto buf_count = graph.get_frames_count();

    for (uint32_t i = 0; i < buf_count; i++) {
      //gpu::Buffer buf;
      //buf.create(VMA_MEMORY_USAGE_CPU_TO_GPU, MAX_TRANSFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      auto buf = gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, MAX_TRANSFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      g_transfer_state->transfer_buffers.emplace_back(std::move(buf));
    }

  }
  void close() {
    if (g_transfer_state) {
      delete g_transfer_state;
    }
  }

  void process_requests(rendergraph::RenderGraph &graph) {

    struct Data {
      std::vector<TransferBlock> blocks;
      uint32_t buffer_id;
    };

    if (g_transfer_state->blocks.empty()) {
      return;
    }

    graph.add_task<Data>("BufferUpdate",
      [&](Data &input, rendergraph::RenderGraphBuilder &builder){
        std::swap(input.blocks, g_transfer_state->blocks);
        input.buffer_id = g_transfer_state->buffer_id;

        for (auto id : g_transfer_state->dirty_buffers) {
          builder.transfer_write(id);
        }
      },
      [=](Data &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
        
        auto api_cmd = cmd.get_command_buffer();
        auto src_buffer = g_transfer_state->transfer_buffers[input.buffer_id]->api_buffer(); 
        
        for (const auto &block : input.blocks) {
          
          VkBufferCopy region {
            .srcOffset = block.src_offset,
            .dstOffset = block.dst_offset,
            .size = block.size
          };

          auto dst_buffer = resources.get_buffer(block.dst)->api_buffer();
          vkCmdCopyBuffer(api_cmd, src_buffer, dst_buffer, 1, &region);
        }
      });
    
    g_transfer_state->dirty_buffers.clear();
    g_transfer_state->buffer_id = (g_transfer_state->buffer_id + 1) % g_transfer_state->transfer_buffers.size();
    g_transfer_state->write_offset = 0;
  }
  
  void write_buffer(rendergraph::BufferResourceId id, uint64_t offset, uint64_t size, const void *data) {
    g_transfer_state->try_upload(id, offset, size, data);
  }



}