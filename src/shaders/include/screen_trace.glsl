#ifndef SCREEN_TRACE_GLSL_INCLUDED
#define SCREEN_TRACE_GLSL_INCLUDED

#define MAX_T_FLOAT 3.402823466e+38

#include "gbuffer_encode.glsl"

void initial_advance_ray(vec3 origin, vec3 dir, vec3 inv_dir, vec2 mip_res, vec2 inv_mip_res, vec2 floor_offset, vec2 uv_offset, out vec3 pos, out float current_t) {
  vec2 cur_pos = mip_res * origin.xy;
  vec2 xy_plane = floor(cur_pos) + floor_offset;
  xy_plane = xy_plane * inv_mip_res + uv_offset;
  vec2 t = (xy_plane - origin.xy) * inv_dir.xy;
  current_t = min(t.x, t.y);
  pos = origin + current_t * dir;
}

bool advance_ray(vec3 origin, vec3 direction, vec3 inv_direction, vec2 current_mip_position, vec2 current_mip_resolution_inv, vec2 floor_offset, vec2 uv_offset, float surface_z, inout vec3 position, inout float current_t) {
  vec2 xy_plane = floor(current_mip_position) + floor_offset;
  xy_plane = xy_plane * current_mip_resolution_inv + uv_offset;
  vec3 boundary_planes = vec3(xy_plane, surface_z);

  // Intersect ray with the half box that is pointing away from the ray origin.
  // o + d * t = p' => t = (p' - o) / d
  vec3 t = (boundary_planes - origin) * inv_direction;

    // Prevent using z plane when shooting out of the depth buffer.
  t.z = direction.z > 0 ? t.z : MAX_T_FLOAT;

  // Choose nearest intersection with a boundary.
  float t_min = min(min(t.x, t.y), t.z);

  // Smaller z means closer to the camera.
  bool above_surface = surface_z > position.z;

  // Decide whether we are able to advance the ray until we hit the xy boundaries or if we had to clamp it at the surface.
  bool skipped_tile = t_min != t.z && above_surface;

  // Make sure to only advance the ray if we're still above the surface.
  current_t = above_surface ? t_min : current_t;

  // Advance ray
  position = origin + current_t * direction;

  return skipped_tile;
}

vec2 get_mip_resolution(vec2 screen_dimensions, int mip_level) {
  return screen_dimensions * pow(0.5, float(mip_level));
}

vec3 hierarchical_raymarch(in sampler2D depth_tex, vec3 origin, vec3 direction, int most_detailed_mip, uint max_traversal_intersections, out bool valid_hit) {
  //const vec3 inv_direction = direction != 0 ? 1.0 / direction : vec3(MAX_T_FLOAT);
  
  const vec3 inv_direction = vec3(
    direction.x != 0 ? 1.0 / direction.x : MAX_T_FLOAT,
    direction.y != 0 ? 1.0 / direction.y : MAX_T_FLOAT,
    direction.z != 0 ? 1.0 / direction.z : MAX_T_FLOAT);
  
  // Start on mip with highest detail.
  int current_mip = most_detailed_mip;

    // Could recompute these every iteration, but it's faster to hoist them out and update them.
  vec2 screen_size = textureSize(depth_tex, 0);
  vec2 current_mip_resolution = get_mip_resolution(screen_size, current_mip);
  vec2 current_mip_resolution_inv = 1.0/current_mip_resolution;

    // Offset to the bounding boxes uv space to intersect the ray with the center of the next pixel.
    // This means we ever so slightly over shoot into the next region. 
  vec2 uv_offset = 0.005 * exp2(most_detailed_mip) / screen_size;
  //uv_offset = direction.xy < 0 ? -uv_offset : uv_offset;
  uv_offset.x = direction.x < 0 ? -uv_offset.x : uv_offset.x;
  uv_offset.y = direction.y < 0 ? -uv_offset.y : uv_offset.y;

  // Offset applied depending on current mip resolution to move the boundary to the left/right upper/lower border depending on ray direction.
  //vec2 floor_offset = direction.xy < 0 ? 0 : 1;
  vec2 floor_offset = vec2(direction.x < 0 ? 0 : 1, direction.y < 0 ? 0 : 1);

    // Initially advance ray to avoid immediate self intersections.
  float current_t;
  vec3 position;
  initial_advance_ray(origin, direction, inv_direction, current_mip_resolution, current_mip_resolution_inv, floor_offset, uv_offset, position, current_t);

  bool exit_due_to_low_occupancy = false;
  int i = 0;
  while (i < max_traversal_intersections && current_mip >= most_detailed_mip && !exit_due_to_low_occupancy) {
    vec2 current_mip_position = current_mip_resolution * position.xy;
    float surface_z = texelFetch(depth_tex, ivec2(current_mip_position), current_mip).x;
    bool skipped_tile = advance_ray(origin, direction, inv_direction, current_mip_position, current_mip_resolution_inv, floor_offset, uv_offset, surface_z, position, current_t);
    current_mip += skipped_tile ? 1 : -1;
    current_mip_resolution *= skipped_tile ? 0.5 : 2;
    current_mip_resolution_inv *= skipped_tile ? 2 : 0.5;
    ++i;
    exit_due_to_low_occupancy = false;
    //exit_due_to_low_occupancy = !is_mirror && WaveActiveCountBits(true) <= min_traversal_occupancy;
  }

  valid_hit = (i <= max_traversal_intersections);

  return position;
}

bool simple_raymarch(in sampler2D depth_tex, vec3 start, vec3 end, const int lod, out vec3 out_ray) {
  vec3 delta = end - start;
  vec2 tex_size = textureSize(depth_tex, lod);

  ivec2 pixel_dist = ivec2(tex_size * abs(delta.xy));
  int steps = max(pixel_dist.x, pixel_dist.y);
  vec3 vec_step = delta/steps;
  
  for (int i = 2; i < steps - 1; i++) {
    vec3 p = start + vec_step * i;
    float depth = texelFetch(depth_tex, ivec2(floor(p.xy * tex_size)), lod).x;

    if (p.z - 0.00001 > depth) {
      
      if (p.z - depth > 0.01) {
        return false;  
      }

      out_ray = p;
      return true;
    }
  }

  return false;
}

vec3 simple_find_horizon(in sampler2D depth_tex, vec3 start, vec2 delta, int max_steps, float thikness, float znear, float zfar) {
  
  vec2 hor_point = start.xy;
  float hor_z = start.z;
  float prev_depth = hor_z;

  vec2 tex_size = textureSize(depth_tex, 0);

  ivec2 pixel_dist = ivec2(tex_size * abs(delta));
  int steps = max(pixel_dist.x, pixel_dist.y);
  vec2 vec_step = delta/steps;
  steps = min(steps, max_steps);

  for (int i = 1; i < steps; i++) {
    vec2 p = start.xy + vec_step * i;
    float depth = texelFetch(depth_tex, ivec2(floor(p.xy * tex_size)), 0).x;
    
    if (depth < hor_z) {
      float linear_prev_depth = linearize_depth2(prev_depth, znear, zfar);
      float linear_depth = linearize_depth2(depth, znear, zfar);
      
      if (abs(linear_prev_depth - linear_depth) > thikness) {
        break;
      }
      
      hor_z = depth;
      hor_point = p;
    }

    prev_depth = depth;
  }
  return vec3(hor_point, hor_z);
}

#endif