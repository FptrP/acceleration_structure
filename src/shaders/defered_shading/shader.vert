#version 460 
#include <screen_verts.glsl>

layout (location = 0) out vec2 out_uv;

void main() {
  gl_Position = OUT_VERTEX;
  out_uv = OUT_UV;
}