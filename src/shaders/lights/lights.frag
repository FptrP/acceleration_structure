#version 460

layout (location = 0) in vec3 IN_COLOR;
layout (location = 1) in vec2 IN_UV;
layout (location = 0) out vec3 OUT_COLOR;

void main() {
  float dist = length(IN_UV - vec2(0.5, 0.5));
  if (dist > 0.5)
    discard;
  OUT_COLOR = IN_COLOR;
}