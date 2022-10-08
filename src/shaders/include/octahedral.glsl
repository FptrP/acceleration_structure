#ifndef OCTAHEDRAL_GLSL_INCLUDED
#define OCTAHEDRAL_GLSL_INCLUDED

float o_sign_nz(in float k) {
  return (k >= 0.0) ? 1.0 : -1.0;
}

vec2 o_sign_nz(in vec2 v) {
  return vec2(o_sign_nz(v.x), o_sign_nz(v.y));
}

vec3 o_sign_nz(in vec3 v) {
  return vec3(o_sign_nz(v.x), o_sign_nz(v.y), o_sign_nz(v.z));
}

vec3 oct_to_sphere(vec2 uv) {
  uv = 2.f * (uv - vec2(0.5f, 0.5f));
  vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
  if (v.z < 0.0) {
    v.xy = (1.0 - abs(v.yx)) * o_sign_nz(v.xy);
  }

  return normalize(v);
}

vec2 sphere_to_oct(in vec3 v) {
  float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
  vec2 result = v.xy * (1.0 / l1norm);
  if (v.z < 0.0) {
    result = (1.0 - abs(result.yx)) * o_sign_nz(result.xy);
  }
  return 0.5f * result + vec2(0.5f, 0.5f);
}

vec3 oct_decode(vec2 uv) {
  uv = 2.f * (uv - vec2(0.5f, 0.5f));
  vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
  if (v.z < 0.0) {
    v.xy = (1.0 - abs(v.yx)) * o_sign_nz(v.xy);
  }

  return normalize(v);
}

vec3 oct_center(vec2 uv) {
  uv = 2.f * (uv - vec2(0.5f, 0.5f));
  vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
  if (v.z < 0.0) {
    v.xy = (1.0 - abs(v.yx)) * o_sign_nz(v.xy);
  }

  return normalize(sign(v));
}

vec2 oct_encode(in vec3 v) {
  float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
  vec2 result = v.xy * (1.0 / l1norm);
  if (v.z < 0.0) {
    result = (1.0 - abs(result.yx)) * o_sign_nz(result.xy);
  }
  return 0.5f * result + vec2(0.5f, 0.5f);
}


vec4 sample_octmap(in sampler2D octmap, in vec3 coord) {
  return texture(octmap, sphere_to_oct(coord));
}

float encode_oct_depth(float z, float n, float f) {
  return f/(f-n) + f*n/((-z) * (f - n));
}

float decode_oct_depth(float d, float n, float f) {
  return -n * f / (d * (f - n) - f);
}

#endif