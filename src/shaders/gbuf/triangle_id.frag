#version 460
#include <triangle_id.glsl>

layout (location = 0) flat in uint DRAWCALL_INDEX;
layout (location = 0) out uint OUT_TRIANGLE_ID;

void main() {
  uint id = uint(gl_PrimitiveID);
  OUT_TRIANGLE_ID = pack_triangle_id(DRAWCALL_INDEX, id);
}