#ifndef GBUFFER_ENCODE_GLSL_INCLUDED
#define GBUFFER_ENCODE_GLSL_INCLUDED


float sign_nz(in float k) {
  return (k >= 0.0) ? 1.0 : -1.0;
}

vec2 sign_nz(in vec2 v) {
  return vec2(sign_nz(v.x), sign_nz(v.y));
}

vec3 sign_nz(in vec3 v) {
  return vec3(sign_nz(v.x), sign_nz(v.y), sign_nz(v.z));
}

vec2 encode_normal(in vec3 v) 
{
  float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
  vec2 result = v.xy * (1.0 / l1norm);
  
  if (v.z < 0.0) {
    result = (1.0 - abs(result.yx)) * sign_nz(result.xy);
  }

  return 0.5f * result + vec2(0.5f, 0.5f);
}

vec3 decode_normal(in vec2 uv)
{
  uv = 2.f * uv - vec2(1.f, 1.f);
  vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
  if (v.z < 0.0) {
    v.xy = (1.0 - abs(v.yx)) * sign_nz(v.xy);
  }
  return normalize(v);
}

vec3 sample_gbuffer_normal(in sampler2D normal_tex, in vec2 uv)
{
  vec2 t = texture(normal_tex, uv).xy;
  return decode_normal(t); 
}

vec3 sample_gbuffer_normal_accurate(in sampler2D normal_tex, in vec2 uv)
{
  vec4 u = textureGather(normal_tex, uv, 0);
  vec4 v = textureGather(normal_tex, uv, 1);
  vec2 t = 0.25 * vec2(dot(u, vec4(1, 1, 1, 1)), dot(v, vec4(1, 1, 1, 1)));
  return decode_normal(t); 
}

float linearize_depth2(float d, float n, float f)
{
  return n * f / (d * (f - n) - f);
}

vec3 reconstruct_view_vec(vec2 uv, float d, float fovy, float aspect, float z_near, float z_far)
{
  float tg_alpha = tan(fovy/2);
  float z = linearize_depth2(d, z_near, z_far);

  float xd = 2 * uv.x - 1;
  float yd = 2 * uv.y - 1;

  float x = -(xd) * (z * aspect * tg_alpha);
  float y = -(yd) * (z * tg_alpha);
  return vec3(x, y, z);
}

float encode_depth(float z, float n, float f) {
  return f/(f-n) + f*n/(z * (f - n));
}

vec3 project_view_vec(vec3 v, float fovy, float aspect, float n, float f) {
  float tg_alpha = tan(fovy/2);
  float z = v.z;

  float depth = f/(f-n) + f*n/(z * (f - n));
  float pu = v.x/(- v.z * tg_alpha * aspect);
  float pv = v.y/(-z * tg_alpha);

  return vec3(0.5 * pu + 0.5, 0.5 * pv + 0.5, depth);
}
/*
vec2 project_direction(vec2 v, float fovy, float aspect) {
  float tg_alpha = tan(fovy/2);
  float pu = v.x/(-tg_alpha * aspect);
  float pv = v.y/(-tg_alpha);
  return normalize(vec2(0.5 * pu + 0.5, 0.5 * pv + 0.5));
}*/

vec3 clip_screen(vec3 start, vec3 end) {
  vec3 delta = normalize(end - start);
  float t = dot(end - start, delta);
  float u_bound = 1e38;
  float v_bound = 1e38;

  
  if (abs(delta.x) > 0.00001)
    u_bound = max((1 - start.x)/delta.x, -start.x/delta.x);
  if (abs(delta.y) > 0.00001)
    v_bound = max((1 - start.y)/delta.y, -start.y/delta.y); 
  
  float t_bound = min(t, min(u_bound, v_bound));
  return start + t_bound * delta;
}

vec2 clip_screen(vec2 start, vec2 end) {
  vec2 delta = normalize(end - start);
  float t = dot(end - start, delta);
  float u_bound = 1e38;
  float v_bound = 1e38;

  
  if (abs(delta.x) > 0.00001)
    u_bound = max((1 - start.x)/delta.x, -start.x/delta.x);
  if (abs(delta.y) > 0.00001)
    v_bound = max((1 - start.y)/delta.y, -start.y/delta.y); 
  
  float t_bound = min(t, min(u_bound, v_bound));
  return start + t_bound * delta;
}

vec2 extend_direction(vec2 start, vec2 delta) {
  float u_bound = 1e38;
  float v_bound = 1e38;

  if (abs(delta.x) <= 0.00001 && abs(delta.y) <= 0.00001) {
    return delta;
  }
  
  if (abs(delta.x) > 0.00001)
    u_bound = max((1 - start.x)/delta.x, -start.x/delta.x);
  if (abs(delta.y) > 0.00001)
    v_bound = max((1 - start.y)/delta.y, -start.y/delta.y); 
  
  float t_bound = min(u_bound, v_bound);
  return start + (t_bound - 0.001) * delta;
}

#endif