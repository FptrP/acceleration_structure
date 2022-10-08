#version 460
#include <gbuffer_encode.glsl>

#define SAMPLE_COUNT 16

layout (location = 0) in vec2 screen_uv;
layout (location = 0) out float occlusion;

layout (set = 0, binding = 0) uniform sampler2D depth;

layout (set = 0, binding = 1) uniform SSAOParams {
  mat4 projection;
  float fovy;
  float aspect;
  float znear;
  float zfar;

  vec3 samples[SAMPLE_COUNT];
};

void main() {
  float frag_depth = texture(depth, screen_uv).r;
  vec3 camera_pos = reconstruct_view_vec(screen_uv, frag_depth, fovy, aspect, znear, zfar);
  
  float sum = 0.f;

  for (uint i = 0; i < SAMPLE_COUNT; i++) {
    vec3 pos = camera_pos + 0.05 * samples[i];
    vec4 ndc = projection * vec4(pos, 1);
    ndc /= ndc.w;

    vec2 sample_uv = 0.5 * (ndc.xy) + vec2(0.5, 0.5);
    float sample_depth = texture(depth, sample_uv).r;
    float pos_depth = ndc.z;

    sum += (pos_depth < sample_depth + 0.0000001) ? 1.0 : 0.0; 
  }

  sum /= SAMPLE_COUNT;

  occlusion = sum;
}