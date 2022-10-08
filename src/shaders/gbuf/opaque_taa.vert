#version 460 core

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_norm;
layout (location = 2) in vec2 in_uv;

layout (set = 0, binding = 0) uniform GbufConst {
  mat4 view_projection;
  mat4 prev_view_projection;
  vec4 jitter;
  vec4 fovy_aspect_znear_zfar;
};

struct Transform {
  mat4 model;
  mat4 normal;
};

layout (std430, set = 0, binding = 1) readonly buffer TransformBuffer {
  Transform transforms[];
};

layout (location = 0) out vec3 out_normal;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out vec4 pos_after;
layout (location = 3) out vec4 pos_before;

layout (push_constant) uniform push_data {
  uint transform_index;
  uint albedo_index;
  uint mr_index;
  uint flags;
};

void main() {
  out_normal = normalize(vec3(transforms[transform_index].normal * vec4(in_norm, 0)));
  out_uv = in_uv;

  vec4 out_vector = view_projection * transforms[transform_index].model * vec4(in_pos, 1); 
  gl_Position = out_vector + out_vector.w * vec4(jitter.xy, 0, 0);

  pos_after = out_vector;
  pos_before = prev_view_projection * transforms[transform_index].model * vec4(in_pos, 1);
}