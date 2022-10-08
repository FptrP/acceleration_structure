#include "framebuffers.hpp"

#include <iostream>

namespace gpu {

  bool FramebufferState::operator==(const FramebufferState &state) const {
    return
      width == state.width &&
      height == state.height &&
      layers == state.layers &&
      attachments_count == state.attachments_count &&
      renderpass == state.renderpass &&
      image_ids == state.image_ids &&
      views == state.views;
  }

  VkFramebuffer FramebufferState::create_fb() const {
    std::vector<VkImageView> api_views;
    api_views.reserve(attachments_count);

    for (uint32_t i = 0; i < attachments_count; i++) {
      auto img = acquire_image(image_ids[i]);
      api_views.push_back(img->get_view(views[i]));
    }

    VkFramebufferCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .renderPass = renderpass,
      .attachmentCount = attachments_count,
      .pAttachments = api_views.data(),
      .width = width,
      .height = height,
      .layers = layers
    };

    VkFramebuffer handle {nullptr};
    VKCHECK(vkCreateFramebuffer(internal::app_vk_device(), &info, nullptr, &handle));
    return handle;
  }

  VkFramebuffer FramebuffersCache::get_framebuffer(FramebufferState &state) {
    auto it = framebuffers.find(state);
    if (it != framebuffers.end()) {
      it->second.last_frame = frame_index;
      return it->second.handle;
    }

    auto &res = framebuffers[state];
    res.handle = state.create_fb();
    res.last_frame = frame_index;
    return res.handle;
  }

  void FramebuffersCache::flip() {
    frame_index = (frame_index + 1) % frames_to_collect;
    for (auto it = framebuffers.begin(); it != framebuffers.end(); it++) {
      if (it->second.last_frame == frame_index) {
        it = framebuffers.erase(it);
      }
    }
  }

}