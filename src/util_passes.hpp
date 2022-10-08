#ifndef UTIL_PASSES_HPP_INCLUDED
#define UTIL_PASSES_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene/camera.hpp"

void gen_perlin_noise2D(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, uint32_t mip, uint32_t layer);
void gen_mipmaps(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image);
void clear_depth(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, float val = 1.0);
void clear_color(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, VkClearColorValue val);
void blit_image(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId src, rendergraph::ImageResourceId dst);

#endif