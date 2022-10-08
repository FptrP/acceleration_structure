#ifndef BACKBUFF_SUBPASS2_HPP_INCLUDED
#define BACKBUFF_SUBPASS2_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene/camera.hpp"
#include "gpu/gpu.hpp"

enum class DrawTex {
  ShowAll = 0,
  ShowR = 1,
  ShowG = 2,
  ShowB = 4,
  ShowA = 8
};


void add_backbuffer_subpass(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId draw_img, VkSampler sampler, DrawTex flags = DrawTex::ShowAll);
void add_backbuffer_subpass(rendergraph::RenderGraph &graph, gpu::ImagePtr &image, VkSampler sampler, DrawTex flags = DrawTex::ShowAll);
void add_present_subpass(rendergraph::RenderGraph &graph);
#endif