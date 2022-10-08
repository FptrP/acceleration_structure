#version 460

layout (location = 0) in vec2 screen_uv;
layout (set = 0, binding = 0) uniform sampler2D PROBE_DEPTH;
layout (location = 0) out float out_depth;


void main() {
  float min_depth = 1000.f;

  //vec2 size = textureSize(depth_tex, 0);
  ivec2 pixel = 2 * ivec2(gl_FragCoord.xy);
  ivec2 size = textureSize(PROBE_DEPTH, 0);

  min_depth = min(texelFetch(PROBE_DEPTH, min(pixel + ivec2(0, 0), size), 0).x, min_depth);
  min_depth = min(texelFetch(PROBE_DEPTH, min(pixel + ivec2(0, 1), size), 0).x, min_depth);
  min_depth = min(texelFetch(PROBE_DEPTH, min(pixel + ivec2(1, 0), size), 0).x, min_depth);
  min_depth = min(texelFetch(PROBE_DEPTH, min(pixel + ivec2(1, 1), size), 0).x, min_depth);

  out_depth = min_depth;
}