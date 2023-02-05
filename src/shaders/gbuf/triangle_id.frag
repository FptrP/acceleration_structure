#version 460
#include <triangle_id.glsl>

layout (location = 0) flat in uint TRIANGLE_ID;
layout (location = 0) out uint OUT_TRIANGLE_ID;

void main() {
  uint id = uint(gl_PrimitiveID);
  OUT_TRIANGLE_ID = set_triangle_id(TRIANGLE_ID, id);
}