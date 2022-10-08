#version 460

layout (location = 0) in vec2 screen_uv;

layout (set = 0, binding = 0) uniform sampler2D DEPTH_TEX;

void main() {
  ivec2 pixel = 2 * ivec2(gl_FragCoord.xy);
  float d0 = texelFetch(DEPTH_TEX, pixel + ivec2(0, 0), 0).x;
  float d1 = texelFetch(DEPTH_TEX, pixel + ivec2(1, 0), 0).x;
  float d2 = texelFetch(DEPTH_TEX, pixel + ivec2(0, 1), 0).x;
  float d3 = texelFetch(DEPTH_TEX, pixel + ivec2(1, 1), 0).x;

  gl_FragDepth = min(min(d0, d1), min(d2, d3));
}