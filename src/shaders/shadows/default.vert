#version 460 core

layout (location = 0) in vec3 in_pos;

layout (set = 0, binding = 0) uniform ShadowConst {
  mat4 mvp;
};

struct Transform {
  mat4 model;
  mat4 normal;
};

layout (std430, set = 0, binding = 1) readonly buffer TransformBuffer {
  Transform transforms[];
};

layout (push_constant) uniform push_data {
  uint transform_index;
};

void main() {
  gl_Position = mvp * transforms[transform_index].model * vec4(in_pos, 1);
}