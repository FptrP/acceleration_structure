#version 460 core
#include <gbuffer_encode.glsl>

layout (location = 0) in vec2 in_uv;
layout (location = 1) in vec3 in_pos;
//layout (location = 1) in vec3 in_normal;

layout (location = 0) out vec4 out_albedo;
layout (location = 1) out float out_distance;

//layout (location = 1) out vec4 out_normal;
//layout (location = 2) out vec4 out_material;

layout (set = 0, binding = 2) uniform texture2D material_textures[64];
layout (set = 0, binding = 3) uniform sampler main_sampler;

layout (push_constant) uniform push_data {
  uint transform_index;
  uint albedo_index;
};

void main() {
  out_albedo = texture(sampler2D(material_textures[albedo_index], main_sampler), in_uv);
  
  if (out_albedo.a == 0) {
    discard;
  }

  out_distance = length(in_pos);
}