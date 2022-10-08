#version 460 core
#include <gbuffer_encode.glsl>

#extension GL_EXT_nonuniform_qualifier : enable 
layout (location = 0) in vec3 in_normal;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec4 pos_after;
layout (location = 3) in vec4 pos_before;

layout (location = 0) out vec4 out_albedo;
layout (location = 1) out vec4 out_normal;
layout (location = 2) out vec4 out_material;
layout (location = 3) out vec2 velocity_vector;

layout (set = 1, binding = 0) uniform sampler2D material_textures[];

layout (push_constant) uniform push_data {
  uint transform_index;
  uint albedo_index;
  uint mr_index;
  uint flags;
};

#define INVALID_INDEX (~0u)

void main() {
  if (albedo_index != INVALID_INDEX) {
    out_albedo = texture(material_textures[albedo_index], in_uv);
  } else {
    out_albedo = vec4(0.5, 0.5, 0.5, 1.0);
  }
  
  if (out_albedo.a == 0) {
    discard;
  }

  out_normal = vec4(encode_normal(in_normal), 0, 0);

  if (mr_index != INVALID_INDEX) {
    out_material = texture(material_textures[mr_index], in_uv);
  } else {
    out_material = vec4(0.5, 0.9, 0.5, 0.5);
  }
  
  velocity_vector = 0.5 * (pos_before.xy/pos_before.w - pos_after.xy/pos_after.w);
}