#ifndef GPU_FRAMEBUFFERS_HPP_INCLUDED
#define GPU_FRAMEBUFFERS_HPP_INCLUDED

#include "driver.hpp"
#include "managed_resources.hpp"
#include "pipelines.hpp"

namespace gpu {

  constexpr uint32_t MAX_ATTACHMENTS = 16; 

  struct FramebufferState {
    FramebufferState() {
      image_ids.reserve(MAX_ATTACHMENTS);
      views.reserve(MAX_ATTACHMENTS);
    }

    bool set_width(uint32_t w) {
      bool modified = width != w;
      width = w;
      dirty |= w;
      return modified;
    }

    bool set_height(uint32_t h) {
      bool mod = height != h;
      height = h;
      dirty |= mod;
      return mod;
    }
  
    bool set_layers(uint32_t l) {
      bool mod = layers != l;
      layers = l;
      dirty |= mod;
      return mod;
    }

    bool set_renderpass(GraphicsPipeline &pipeline) {
      uint32_t new_count = pipeline.get_renderpass_desc().formats.size();
      auto new_handle = pipeline.get_renderpass();
      bool mod = new_handle != renderpass || new_count != attachments_count;
      
      renderpass = new_handle;
      attachments_count = new_count;

      //views.resize(attachments_count);
      //image_ids.resize(attachments_count);

      dirty |= mod;
      return mod; 
    }

    bool set_attachment(uint32_t index, const ImagePtr &image, ImageViewRange range) {
      if (index >= MAX_ATTACHMENTS)
        throw std::runtime_error {"Too many attachments"};

      if (index >= image_ids.size()) {
        image_ids.resize(index + 1);
        views.resize(index + 1);
      }

      auto id = image.get_id();
      auto &src_id = image_ids.at(index);
      auto &src_view = views.at(index);
      bool mod = id != src_id || range != src_view;
      
      src_view = range;
      src_id = id;
      
      dirty |= mod;
      return mod;
    }

    bool is_dirty() const { return dirty; }

    size_t get_hash() const {
      if (!dirty)
        return hash;
      
      hash = 0;
      
      hash_combine(hash, width);
      hash_combine(hash, height);
      hash_combine(hash, layers);
      hash_combine(hash, renderpass);

      for (uint32_t i = 0; i < attachments_count; i++) {
        hash_combine(hash, views[i]);
        hash_combine(hash, image_ids[i]);
      }

      dirty = false;
      return hash;
    } 

    uint32_t get_width() const { return width; }
    uint32_t get_height() const { return height; }
    
    VkFramebuffer create_fb() const;
    bool operator==(const FramebufferState &st) const;
  private:
    mutable bool dirty = true;
    mutable size_t hash = 0; 

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t layers = 1;
    
    uint32_t attachments_count = 0;
    VkRenderPass renderpass {nullptr};
    
    std::vector<ImageViewRange> views;
    std::vector<DriverResourceID> image_ids;
  };
}

namespace std {
  template <>
  struct hash<gpu::FramebufferState> {
    size_t operator()(const gpu::FramebufferState &key) const {
      return key.get_hash();
    }
  };
}

namespace gpu {

  struct FramebuffersCache {
    FramebuffersCache(uint32_t frames_to_collect_) : frames_to_collect {frames_to_collect_} {}
    ~FramebuffersCache() { framebuffers.clear(); }

    VkFramebuffer get_framebuffer(FramebufferState &state);
    void flip();

    FramebuffersCache(const FramebuffersCache&) = delete;
    FramebuffersCache &operator=(const FramebuffersCache&) = delete;
  private:

    struct ApiFramebuffer {
      ApiFramebuffer() {}
      ~ApiFramebuffer() { if (handle) vkDestroyFramebuffer(internal::app_vk_device(), handle, nullptr); }

      VkFramebuffer handle {nullptr};
      uint32_t last_frame = 0;
    };
    
    uint32_t frames_to_collect;
    uint32_t frame_index = 0;
    std::unordered_map<FramebufferState, ApiFramebuffer> framebuffers; 
  };

}


#endif