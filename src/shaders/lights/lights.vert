#version 460 

#define MAX_LIGHTS 5

layout (push_constant) uniform PushConstants {
  mat4 projection;
  vec4 camera_position;
  vec4 color;
};

const float QUAD_SIZE = 0.05;
const vec2 VERT_OFFSETS[] = vec2[](
  vec2( QUAD_SIZE,  QUAD_SIZE),
  vec2( QUAD_SIZE, -QUAD_SIZE),
  vec2(-QUAD_SIZE, -QUAD_SIZE),

  vec2(-QUAD_SIZE,  QUAD_SIZE),
  vec2( QUAD_SIZE,  QUAD_SIZE),
  vec2(-QUAD_SIZE, -QUAD_SIZE)
);

const vec2 VERT_UV[] = vec2[](
  vec2(1, 1),
  vec2(1, 0),
  vec2(0, 0),

  vec2(0, 1),
  vec2(1, 1),
  vec2(0, 0)
);

layout (location = 0) out vec3 OUT_COLOR;
layout (location = 1) out vec2 OUT_UV;

void main() {
  vec4 offset = vec4(VERT_OFFSETS[gl_VertexIndex], 0, 0);
  gl_Position = projection * (camera_position + offset);
  OUT_COLOR = color.rgb;
  OUT_UV = VERT_UV[gl_VertexIndex];
}