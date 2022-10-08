#version 460

layout(set = 0, binding = 0) uniform sampler2D GBUFFER_DEPTH;
layout(set = 0, binding = 1) uniform sampler2D GBUFFER_NORMAL;
layout(set = 0, binding = 2) uniform sampler2D GBUFFER_VELOCITY;

layout (location = 0) in vec2 screen_uv;

layout (location = 0) out vec2 out_normal; 
layout (location = 1) out vec2 out_velocity;

void main() {
  float min_depth = 1000.f;

  //vec2 size = textureSize(depth_tex, 0);
  ivec2 pixel = 2 * ivec2(gl_FragCoord.xy);

  float d0 = texelFetch(GBUFFER_DEPTH, pixel + ivec2(0, 0), 0).x;
  float d1 = texelFetch(GBUFFER_DEPTH, pixel + ivec2(1, 0), 0).x;
  float d2 = texelFetch(GBUFFER_DEPTH, pixel + ivec2(0, 1), 0).x;
  float d3 = texelFetch(GBUFFER_DEPTH, pixel + ivec2(1, 1), 0).x;

  ivec2 normal_offset = ivec2(0, 0); 
  min_depth = min(min(d0, d1), min(d2, d3));
  
  if (min_depth == d1) {
    normal_offset = ivec2(1, 0);
  } else if (min_depth == d2) {
    normal_offset = ivec2(0, 1);
  } else if (min_depth == d3) {
    normal_offset = ivec2(1, 1);
  }

  out_normal = texelFetch(GBUFFER_NORMAL, pixel + normal_offset, 0).xy;
  out_velocity = texelFetch(GBUFFER_VELOCITY, pixel + normal_offset, 0).xy;
  gl_FragDepth = min_depth;
}