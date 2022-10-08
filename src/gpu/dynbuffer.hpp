#ifndef DYNBUFFER_HPP_INCLUDED
#define DYNBUFFER_HPP_INCLUDED

#include "resources.hpp"
#include "managed_resources.hpp"
#include <stdexcept>

namespace gpu {

  template <typename T>
  struct UboBlock {
    T *ptr;
    uint32_t offset;
  };

  struct UniformBufferPool {
    UniformBufferPool(uint64_t alignment, uint64_t mem_size)
      : buffer {gpu::create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, mem_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)}, mem_alignment {alignment}
    {
    }

    VkBuffer api_buffer() const { return buffer->api_buffer(); }

    template<typename T>
    UboBlock<T> allocate_ubo() {
      auto val = allocate_chunk(sizeof(T));
      return {static_cast<T*>(val.ptr), val.offset};
    } 

    void reset() { write_offset = 0; }

    UniformBufferPool(UniformBufferPool &&) = default;
    UniformBufferPool &operator=(UniformBufferPool &&) = default; 
  private:
    gpu::BufferPtr buffer;
    uint64_t write_offset = 0;
    uint64_t mem_alignment = 0;

    UboBlock<void> allocate_chunk(uint64_t mem_size) {
      auto aligned_size = get_block_size(mem_size, mem_alignment);
      auto offset = write_offset;
      auto ptr = static_cast<uint8_t*>(buffer->get_mapped_ptr()) + offset;
      write_offset += aligned_size;
      if (write_offset > buffer->get_size()) {
        throw std::runtime_error {"UBOPool out of memory!\n"};
      }
      return {static_cast<void*>(ptr), uint32_t(offset)};
    }

    static uint64_t get_block_size(const uint64_t size, const uint64_t alignment) {
      if (!alignment) {
        return size;
      }

      auto mod = size % alignment;
      if (mod) {
        return size + alignment - mod;
      }
      return size;
    }

  };

}


#endif