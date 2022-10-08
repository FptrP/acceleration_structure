#ifndef IMGUI_PASS_HPP_INCLUDED
#define IMGUI_PASS_HPP_INCLUDED

#include "gpu/gpu.hpp"
#include "gpu/imgui_context.hpp"

void imgui_init(SDL_Window *wdinow, VkRenderPass renderpass);
void imgui_handle_event(const SDL_Event &event);
void imgui_draw(VkCommandBuffer cmd);
void imgui_create_fonts(gpu::TransferCmdPool &transfer_pool);
void imgui_new_frame();
void imgui_close();

#endif