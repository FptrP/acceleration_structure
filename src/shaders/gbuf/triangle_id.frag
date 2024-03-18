#version 460
#include "../include/triangle_id.glsl"

#extension GL_EXT_nonuniform_qualifier : enable

layout (location = 0) flat in uint DRAWCALL_INDEX;
layout (location = 1) in vec2 IN_UV;

layout (location = 0) out uint OUT_TRIANGLE_ID;

layout (push_constant) uniform PushConstants {
  uint transform_index;
  uint drawcall_index;
  uint alpha_tex_index;
};

layout (set = 1, binding = 0) uniform sampler2D material_textures[];

void main() {
  if (alpha_tex_index != 0xffffffff) {
    float alpha = texture(material_textures[alpha_tex_index], IN_UV).a;
    if (alpha < 0.5f)
      discard;
  }

  uint id = uint(gl_PrimitiveID);
  OUT_TRIANGLE_ID = pack_triangle_id(DRAWCALL_INDEX, id);
}