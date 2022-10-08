#version 460 core

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_norm;
layout (location = 2) in vec2 in_uv;

layout (set = 0, binding = 0) uniform GbufConst {
  mat4 projection;
  mat4 camera;
};

struct Transform {
  mat4 model;
  mat4 normal;
};

layout (std430, set = 0, binding = 1) readonly buffer TransformBuffer {
  Transform transforms[];
};

layout (location = 0) out vec2 out_uv;
layout (location = 1) out vec3 out_pos;
//layout (location = 1) out vec3 out_normal;


layout (push_constant) uniform push_data {

  uint transform_index;
  uint albedo_index;
};

void main() {
  //out_normal = normalize(vec3(transforms[transform_index].normal * vec4(in_norm, 0)));
  vec4 view_pos = camera * transforms[transform_index].model * vec4(in_pos, 1);
  out_uv = in_uv;
  out_pos = view_pos.xyz; 
  gl_Position = projection * view_pos;
}