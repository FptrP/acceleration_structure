#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

#include <gbuffer_encode.glsl>

#define PI 3.1415926535897932384626433832795

layout (location = 0) in vec2 screen_uv;
layout (location = 0) out vec4 occlusion;

layout (set = 0, binding = 0) uniform GTAORTParams {
  mat4 camera_to_world;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};
const int DIRECTION_COUNT = 64; 

layout (set = 0, binding = 1) uniform sampler2D depth;
layout (set = 0, binding = 2) uniform sampler2D gbuffer_normal;
layout (set = 0, binding = 3) uniform accelerationStructureEXT tlas;

layout (set = 0, binding = 4) uniform RandomVectors {
  vec4 ao_directions[DIRECTION_COUNT];
};

layout (push_constant) uniform PushConsts {
  float rotation;
};


float get_visibility(in vec3 world_pos, in vec3 dir) {
  rayQueryEXT ray_query;
	rayQueryInitializeEXT(ray_query, tlas, 0, 0xFF, world_pos, 1e-12, dir, 1.0);
  while (rayQueryProceedEXT(ray_query)) {}

  float visibility = 1.f;
  if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
    visibility = 0.f;
  }

  return visibility;
}

vec3 get_tangent(in vec3 n) {
  float max_xy = max(abs(n.x), abs(n.y));
  vec3 t;
  
  if (max_xy < 0.00001) {
    t = vec3(1, 0, 0);
  } else {
    t = vec3(n.y, -n.x, 0);
  }

  return normalize(t);
}

float gtao_direction(in ivec2 pos) {
  return (1.0 / 16.0) * ((((pos.x + pos.y) & 3) << 2) + (pos.x & 3));
}

void main() {
  ivec2 pixel_pos = ivec2(gl_FragCoord.xy);

  float frag_depth = texture(depth, screen_uv).r;
  if (frag_depth >= 1.f) {
    occlusion = vec4(0.f, 1.f, 0.f, 0.f);
    return;
  }
  
  vec3 view_vec = reconstruct_view_vec(screen_uv, frag_depth, fovy, aspect, znear, zfar);
  vec3 world_pos = (camera_to_world * vec4(view_vec, 1)).xyz;
  
  vec3 normal = decode_normal(texture(gbuffer_normal, screen_uv).xy);
  world_pos += 1e-6 * normal;

  vec3 tangent = get_tangent(normal);
  vec3 bitangent = normalize(cross(normal, tangent));
  tangent = normalize(cross(bitangent, normal));
  
  float angle = 2 * PI * (rotation + gtao_direction(pixel_pos));
  tangent = normalize(cos(angle) * tangent + sin(angle) * bitangent);
  bitangent = normalize(cross(normal, tangent));
  tangent = normalize(cross(bitangent, normal));

  float sum = 0.f;
  const int SAMPLES_COUNT = DIRECTION_COUNT;
  for (int i = 0; i < SAMPLES_COUNT; i++) {
    vec3 dir = normalize(ao_directions[i].xyz);
    dir = normalize(dir.z * normal + dir.x * tangent + dir.y * bitangent);
    vec3 scaled_dir = 0.2 * dir; 

    sum += get_visibility(world_pos, scaled_dir) * max(dot(dir, normal), 0.f);
  }

  sum /= SAMPLES_COUNT;
  occlusion = vec4(2 * sum, 1.0, 0.0, 0.0); //1/PI * 2 * PI * R^2
}