#version 460
#include <gbuffer_encode.glsl>
#include <screen_trace.glsl>
#include <brdf.glsl>
layout (location = 0) in vec2 screen_uv;

layout (set = 0, binding = 0) uniform sampler2D normal_tex;
layout (set = 0, binding = 1) uniform sampler2D depth_tex;
layout (set = 0, binding = 2) uniform sampler2D frame_tex;

layout (set = 0, binding = 3) uniform SSRParams {
  mat4 camera_normal;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

layout (set = 0, binding = 4) uniform sampler2D material_tex;


layout (location = 0) out vec4 out_reflection;

bool simple_raymarch(in sampler2D depth_tex, vec3 start, vec3 end, const int lod, out vec3 out_ray);
bool hiz_trace(in sampler2D depth_tex, vec3 start, vec3 end, out vec3 out_ray);

void main() {
  vec2 tex_size = textureSize(frame_tex, 0);
  vec2 aligned_screen_uv = floor(screen_uv*tex_size)/tex_size + 0.5/tex_size;
  
  vec3 material = texture(material_tex, screen_uv).rgb;
  const float metallic = material.b;
  const float roughness = material.g;
  //roughness = mix(0.9, 1.0, roughness);
  //roughness = 1.0;
  float pixel_depth = texture(depth_tex, aligned_screen_uv).x;
  vec3 pixel_normal_world = sample_gbuffer_normal(normal_tex, aligned_screen_uv);
  vec3 pixel_normal = normalize((camera_normal * vec4(pixel_normal_world, 0)).xyz);
  
  vec3 view_vec = reconstruct_view_vec(aligned_screen_uv, pixel_depth, fovy, aspect, znear, zfar);

  vec3 R = reflect(view_vec, pixel_normal);
  //vec3 H = normalize(-normalize(view_vec) + normalize(R));
  vec3 H = pixel_normal;

  vec3 start = project_view_vec(view_vec + 0.0005 * pixel_normal, fovy, aspect, znear, zfar);
  vec3 p = project_view_vec(view_vec + R, fovy, aspect, znear, zfar);
  vec3 delta = normalize(p - start);

  if (abs(delta.z) < 0.0000001) {
    out_reflection = vec4(0, 0, 0, 0);
    return;  
  }

  float t_bound = (1-start.z)/delta.z;
  float u_bound = max((1 - start.x)/delta.x, -start.x/delta.x);
  float v_bound = max((1 - start.y)/delta.y, -start.y/delta.y); 
  t_bound = min(t_bound, min(u_bound, v_bound));

  vec3 end = start + t_bound * delta;
  delta = end - start;

  const int LOD = 2;
  vec3 out_ray;
  bool valid_hit;

  out_ray = hierarchical_raymarch(depth_tex, start, end - start, 0, 100, valid_hit);
  if (valid_hit)
  {
    vec2 screen_size = textureSize(frame_tex, 0);
    vec2 dist0 = abs(out_ray.xy - start.xy);
    vec2 min_dist = vec2(2)/screen_size;

    if (dist0.x < min_dist.x && dist0.y < min_dist.y) {
      out_reflection = vec4(0, 0, 0, 0);
      return;
    }

    vec3 hit_normal_world = sample_gbuffer_normal(normal_tex, out_ray.xy);
    vec3 hit_normal = (camera_normal * vec4(hit_normal_world, 0)).xyz;

    if (dot(hit_normal, R) > 0) {
      out_reflection = vec4(0, 0, 0, 0);
      return;
    }

    float hit_depth = textureLod(depth_tex, out_ray.xy, 0).x;
    if (out_ray.z > hit_depth + 0.0001) {
      out_reflection = vec4(0, 0, 0, 0);
      return; 
    }

    vec2 fov = 0.05 * vec2(screen_size.y / screen_size.x, 1);
    vec2 border = smoothstep(vec2(0), fov, out_ray.xy) * (1 - smoothstep(vec2(1 - fov), vec2(1), out_ray.xy));
    float coef = border.x * border.y;
    
    out_reflection = coef * texture(frame_tex, out_ray.xy) * DistributionGGX(pixel_normal, H, roughness) * max(dot(pixel_normal, R), 0);
    return;
  }

  out_reflection = vec4(0, 0, 0, 0);
}

vec2 get_cell(vec2 ray, vec2 cell_count) {
  return floor(ray * cell_count);
}

vec2 cell_count(int level) {
  return vec2(textureSize(depth_tex, level));
}

vec3 intersect_cell_boundary(vec3 pos, vec3 dir, vec2 cell_id, vec2 cell_count, vec2 cross_step, vec2 cross_offset) {
  vec2 cell_size = 1.0 / cell_count;
  vec2 planes = cell_id/cell_count + cell_size * cross_step;
  vec2 solutions = (planes - pos.xy)/dir.xy;
  vec3 intersection_pos = pos + dir * min(solutions.x, solutions.y);
  intersection_pos.xy += (solutions.x < solutions.y) ? vec2(cross_offset.x, 0.0) : vec2(0.0, cross_offset.y);
  return intersection_pos;
}

bool crossed_cell_boundary(vec2 cell_id_one, vec2 cell_id_two) {
 return int(cell_id_one.x) != int(cell_id_two.x) || int(cell_id_one.y) != int(cell_id_two.y);
}

float minimum_depth_plane(vec2 ray, int level, vec2 cell_count) {
 return texelFetch(depth_tex, ivec2(floor(ray * cell_count)), level).x;
}

const int MAX_LOD = 8;
const uint MAX_ITERATIONS = 80; 

bool hiz_trace(in sampler2D depth_tex, vec3 start, vec3 end, out vec3 out_ray) {
  vec3 ray = start;
  vec3 v = end - start;
  vec3 v_z = v/v.z;

  if (v.z <= 0) return false;

  vec2 cross_step = vec2(v.x >= 0? 1.0 : -1.0, v.y >= 0? 1.0 : -1.0);
  vec2 cross_offset = cross_step * 0.0001;
  cross_step = clamp(cross_step, 0, 1);

  int level = 0;
  vec2 hi_z_size = cell_count(level);
  vec2 ray_cell = get_cell(ray.xy, hi_z_size);
  ray = intersect_cell_boundary(ray, v, ray_cell, hi_z_size, cross_step, cross_offset);

  uint iterations = 0;
  while (level >= 0 && iterations < MAX_ITERATIONS) {
    vec2 current_cell_count = cell_count(level);
    vec2 old_cell_id = get_cell(ray.xy, current_cell_count);

    float min_z = minimum_depth_plane(ray.xy, level, current_cell_count);

    float min_minus_ray = min_z - ray.z;    
    vec3 temp_ray = (min_minus_ray > 0)? (ray + v_z * min_minus_ray) : ray;
    vec2 new_cell_id = get_cell(temp_ray.xy, current_cell_count);

    if (crossed_cell_boundary(old_cell_id, new_cell_id)) {
      temp_ray = intersect_cell_boundary(ray, v, old_cell_id, current_cell_count, cross_step, cross_offset);
      level = min(MAX_LOD, level + 2);
    }

    ray = temp_ray;
    level -= 1;
    iterations++;
  }

  out_ray = ray;
  return (level < 0);
}