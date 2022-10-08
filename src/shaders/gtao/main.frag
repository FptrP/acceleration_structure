#version 460
#include <gbuffer_encode.glsl>

layout (location = 0) in vec2 screen_uv;
layout (location = 0) out float occlusion;

layout (set = 0, binding = 0) uniform sampler2D depth;

layout (set = 0, binding = 1) uniform GTAOParams {
  mat4 normal_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

layout (set = 0, binding = 2) uniform sampler2D gbuffer_normal;

layout (push_constant) uniform PushConstants {
  float angle_offset;
};

#define USE_SAMPLES_TRACE 1
#define START_X 0.5 - 1e-6
#define START_Y 0.5 - 1e-6
#define END_X 0.5 + 8.0/1920.0
#define END_Y 0.5 + 4.0/1920.0
#include "trace_samples.glsl"

#define RADIUS 0.01
#define SAMPLES 20
#define THIKNESS 0.05

#define UNIFORM_AO 0
#define COS_AO 1

#define AO_MODE COS_AO

#define PI 3.1415926535897932384626433832795

float gtao_direction(in ivec2 pos);
float gtao_direction_2x2(in ivec2 pos);
float gtao_normal_space(in ivec2 pos, in vec2 screen_uv, uint dirs_count);
float gtao_camera_space(in ivec2 pos, in vec2 screen_uv, uint dirs_count);


void main() {
  ivec2 pixel_pos = ivec2(gl_FragCoord.xy);
  occlusion = gtao_camera_space(pixel_pos, screen_uv, 1);
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

const float MAX_THIKNESS = 0.1;

float find_horizon(in vec2 start, in vec3 camera_start, in vec2 dir, int samples_count, in vec3 v) {
  float h_cos = -1.0;
  float previous_z = camera_start.z;

  for (int i = 1; i <= samples_count; i++) {
    vec2 tc = start + (float(i)/samples_count) * dir;
    float sample_depth = textureLod(depth, tc, 0).r;
    TRACE_SAMPLE_UV(start, tc);

    vec3 sample_pos = reconstruct_view_vec(tc, sample_depth, fovy, aspect, znear, zfar);
    
    if (sample_pos.z > previous_z + MAX_THIKNESS) {
      break;
    }

    previous_z = sample_pos.z;
    vec3 sample_offset = sample_pos - camera_start;

    float sample_cos = dot(v, normalize(sample_offset));
    if (sample_cos > h_cos) {
      h_cos = sample_cos;
    }
  }
  return h_cos;
}

float find_horizon_pos(in vec2 start, in vec3 camera_start, in vec2 dir, int samples_count, in vec3 v, inout vec3 delta) {
  float h_cos = -1.0;
  float previous_z = camera_start.z;

  for (int i = 1; i <= samples_count; i++) {
    vec2 tc = start + (float(i)/samples_count) * dir;
    float sample_depth = textureLod(depth, tc, 0).r;
    TRACE_SAMPLE_UV(start, tc);

    vec3 sample_pos = reconstruct_view_vec(tc, sample_depth, fovy, aspect, znear, zfar);
    
    if (sample_pos.z > previous_z + MAX_THIKNESS) {
      break;
    }

    previous_z = sample_pos.z;
    vec3 sample_offset = sample_pos - camera_start;
    delta = normalize(sample_offset);

    float sample_cos = dot(v, delta);
    if (sample_cos > h_cos) {
      h_cos = sample_cos;
    }
  }
  return h_cos;
}

float gtao_normal_space(in ivec2 pos, in vec2 screen_uv, uint dirs_count) {
  float frag_depth = texture(depth, screen_uv).r;
  if (frag_depth >= 1.f) {
    return 1.f;
  }

  vec3 camera_pos = reconstruct_view_vec(screen_uv, frag_depth, fovy, aspect, znear, zfar);
  vec3 w0 = -normalize(camera_pos);
  
  vec3 camera_normal = normalize((normal_mat * vec4(decode_normal(texture(gbuffer_normal, screen_uv).xy), 0)).xyz);
  vec3 tangent = get_tangent(camera_normal);
  vec3 bitangent = normalize(cross(camera_normal, tangent));
  tangent = normalize(cross(bitangent, camera_normal));

  float base_angle = gtao_direction(pos) + angle_offset; 

  vec2 dir_radius = min(200.0/length(camera_pos), 32.0) / vec2(textureSize(depth, 0));
  
  float sum = 0.f;

  for (int dir_index = 0; dir_index < dirs_count; dir_index++) {
    float angle = 2 * PI * (base_angle + float(dir_index)/float(dirs_count));

    vec3 sample_vector = cos(angle) * tangent + sin(angle) * bitangent;
    vec2 sample_direction = project_view_vec(camera_pos + sample_vector, fovy, aspect, znear, zfar).xy - screen_uv;
    sample_direction = dir_radius * normalize(sample_direction);

    vec3 delta;
    
    float h_cos = find_horizon(screen_uv, camera_pos, sample_direction, SAMPLES, camera_normal);
    
    h_cos = max(h_cos, 0);
    sum += (1 - h_cos * h_cos);
  }

  return sum/float(dirs_count); 
}

float gtao_camera_space(in ivec2 pos, in vec2 screen_uv, uint dirs_count) {
  float frag_depth = texture(depth, screen_uv).r;
  if (frag_depth >= 1.f) {
    return 1.f;
  }

  vec3 camera_pos = reconstruct_view_vec(screen_uv, frag_depth, fovy, aspect, znear, zfar);
  vec3 w0 = -normalize(camera_pos);
  vec3 camera_normal = normalize((normal_mat * vec4(decode_normal(texture(gbuffer_normal, screen_uv).xy), 0)).xyz);

  vec2 dir_radius = min(200.0/length(camera_pos), 32.0) / vec2(textureSize(depth, 0));
  float base_angle = gtao_direction(pos) + angle_offset; 
  float sum = 0.f;

  for (int dir_index = 0; dir_index < dirs_count; dir_index++) {
    float angle = 2 * PI * (base_angle + float(dir_index)/float(dirs_count));

    vec2 sample_direction = dir_radius * vec2(cos(angle), sin(angle));
    vec3 sample_end_pos = reconstruct_view_vec(screen_uv + sample_direction, frag_depth, fovy, aspect, znear, zfar);

    vec3 slice_normal = normalize(cross(w0, -sample_end_pos));
    vec3 normal_projected = camera_normal - dot(camera_normal, slice_normal) * slice_normal;
    float n = PI/2.0 - acos(dot(normalize(normal_projected), normalize(sample_end_pos - camera_pos)));

    float h_cos = find_horizon(screen_uv, camera_pos, sample_direction, SAMPLES, w0);
    float h = acos(h_cos);
    h = min(n + min(h - n, PI/2.0), h);
    sum += length(normal_projected) * 0.25 * max(-cos(2 * h - n) + cos(n) + 2*h*sin(n), 0);
  }

  return 2 * sum/float(dirs_count);
}

float gtao_direction(in ivec2 pos) { // -> full rotation every 4 pixels
  return (1.0 / 16.0) * ((((pos.x + pos.y) & 3) << 2) + (pos.x & 3));
}

float gtao_direction_2x2(in ivec2 pos) {
  return (1.0/4.0) * (((pos.y & 1) << 1) + (pos.x & 1));
}