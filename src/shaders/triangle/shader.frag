#version 460 core

layout (set = 0, binding = 0) uniform FrameD {
  mat4 mvp;
  vec4 color;
};

layout (location = 0) out vec4 frag_color;

void main() {
  frag_color = color;
}