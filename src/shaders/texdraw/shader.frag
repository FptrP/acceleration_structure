#version 460 core
#include <gbuffer_encode.glsl>

layout (set = 0, binding = 0) uniform sampler2D target_tex;

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 out_color;

#define SHOW_ALL 0
#define SHOW_R 1
#define SHOW_G 2
#define SHOW_B 4
#define SHOW_A 8

layout (push_constant) uniform Constants {
  uint flags;
};

void main() {
  vec4 c = texture(target_tex, in_uv);
  out_color = c;
  if ((flags & SHOW_R) != 0) {
    out_color = vec4(c.r, c.r, c.r, c.r);
  }
  if ((flags & SHOW_G) != 0) {
    out_color = vec4(c.g, c.g, c.g, c.g);
  }
  if ((flags & SHOW_B) != 0) {
    out_color = vec4(c.b, c.b, c.b, c.b);
  }
  if ((flags & SHOW_A) != 0) {
    out_color = vec4(c.a, c.a, c.a, c.a);
  }
}