#include "imgui_pass.hpp"

#include <memory> 

gpu::ImguiContext *context {nullptr};

void imgui_init(SDL_Window *window, VkRenderPass renderpass) {
  auto image_count = gpu::app_swapchain().get_images_count();
  context = new gpu::ImguiContext {window, gpu::app_instance(), gpu::app_device(), image_count, renderpass};
}

void imgui_handle_event(const SDL_Event &event) {
  context->process_event(event);
}

void imgui_draw(VkCommandBuffer cmd) {
  context->render(cmd);
}

void imgui_create_fonts(gpu::TransferCmdPool &transfer_pool) {
  context->create_fonts(transfer_pool);
}

void imgui_new_frame() {
  context->new_frame();
}

void imgui_close() {
  delete context;
}