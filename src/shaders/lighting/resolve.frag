#version 460
#include "../include/gbuffer_encode.glsl"
#include "../include/brdf.glsl"

layout (location = 0) in vec2 screen_uv;
layout (location = 0) out vec3 out_color;

layout (set = 0, binding = 0) uniform sampler2D DEPTH_TEX;
layout (set = 0, binding = 1) uniform sampler2D NORMAL_TEX;
layout (set = 0, binding = 2) uniform sampler2D MATERIAl_TEX;

layout (set = 0, binding = 3) uniform sampler2D DIFFUSE_TEX;
layout (set = 0, binding = 4) uniform sampler2D SPECULAR_TEX;

layout (set = 0, binding = 5) uniform sampler2D REFLECTIONS_TEX;

layout (set = 0, binding = 6) uniform Constants {
  mat4 normal_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

layout (push_constant) uniform PushConstants {
  float reflectiveness;
};

void main() {
  vec3 specular = texture(SPECULAR_TEX, screen_uv).xyz;
  vec3 diffuse = texture(DIFFUSE_TEX, screen_uv).xyz;
  vec3 reflection = texture(REFLECTIONS_TEX, screen_uv).xyz;
  vec4 material = texture(MATERIAl_TEX, screen_uv);

  vec3 N = sample_gbuffer_normal(NORMAL_TEX, screen_uv);
  N = normalize(vec3(normal_mat * vec4(N, 0)));

  float depth = texture(DEPTH_TEX, screen_uv).x;

  vec3 view_vec = reconstruct_view_vec(screen_uv, depth, fovy, aspect, znear, zfar);

  vec3 V = normalize(-view_vec);
  vec3 R = reflect(V, N);

  const float metallic = mix(reflectiveness, 1.0, material.b);
  const float roughness = material.g;

  vec3 F0 = F0_approximation(normalize(diffuse), metallic);
  vec3 F = fresnelSchlick(max(dot(N, V), 0.0), F0); 

  reflection = metallic * reflection * F;

  out_color = specular + diffuse + reflection;
}