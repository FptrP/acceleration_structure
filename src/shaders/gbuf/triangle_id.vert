#version 460
#include <triangle_id.glsl>

layout (push_constant) uniform PushConstants {
  uint transform_index;
  uint drawcall_index;
};

layout (set = 0, binding = 0, std430) readonly buffer TransformBuffer {
  Transform TRANSFORMS[];
};

layout (set = 0, binding = 1) uniform GbufConst {
  mat4 view_projection;
  vec4 jitter;
};

layout (location = 0) in vec3 in_pos;

layout (location = 0) flat out uint OUT_DRAWCALL_INDEX;

void main() {
  vec4 pos = vec4(in_pos, 1);
  mat4 transform = TRANSFORMS[transform_index].model;
  vec4 out_vector = view_projection * transform * pos;
  
  gl_Position = out_vector + out_vector.w * vec4(jitter.xy, 0, 0);
  OUT_DRAWCALL_INDEX = drawcall_index;
}