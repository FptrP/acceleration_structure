#ifndef GPU_TRANSFER_HPP_INCLUDED
#define GPU_TRANSFER_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"

namespace gpu_transfer {
  
  constexpr uint64_t MAX_TRANSFER_SIZE = (1 << 20); //1 Mb 

  void init(const rendergraph::RenderGraph &graph);
  void close();
  void process_requests(rendergraph::RenderGraph &graph);
  
  void write_buffer(rendergraph::BufferResourceId id, uint64_t offset, uint64_t size, const void *data);
}


#endif
