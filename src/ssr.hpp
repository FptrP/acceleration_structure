#ifndef SSR_HPP_INCLUDED
#define SSR_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene/camera.hpp"

rendergraph::ImageResourceId create_ssr_tex(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h);

struct SSRParams {
  glm::mat4 normal_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

void add_ssr_pass(
  rendergraph::RenderGraph &graph,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId normal,
  rendergraph::ImageResourceId color,
  rendergraph::ImageResourceId material,
  rendergraph::ImageResourceId out,
  const SSRParams &params);

#endif