#version 460 core
#include <screen_verts.glsl>

/*vec2 verts[] = vec2[](
  vec2(-1, -1),
  vec2(-1, 3),
  vec2(3, -1)
);

vec2 uv_coords[] = vec2[](
  vec2(0, 0),
  vec2(0, 2),
  vec2(2, 0)
);*/

layout (location = 0) out vec2 out_uv;

void main() {
  gl_Position = OUT_VERTEX;
  out_uv = OUT_UV;
}
