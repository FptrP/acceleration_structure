#version 460

layout (location = 0) flat in uint TRIANGLE_ID;
layout (location = 0) out uint OUT_TRIANGLE_ID;

void main() {
  OUT_TRIANGLE_ID = TRIANGLE_ID;
}