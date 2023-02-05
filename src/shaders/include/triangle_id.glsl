#ifndef TRIANGLE_ID_GLSL_INCLUDED
#define TRIANGLE_ID_GLSL_INCLUDED

#define TRIANGLE_BITS 16
#define PRIMITIVE_BITS 8
#define TRANSFORM_BITS 8

#define TRIANGLE_MASK 0x0000FFFFu
#define PRIMITIVE_MASK 0x00FF0000u
#define TRANSFORM_MASK 0xFF000000u 

#define INVALID_TRIANGLE_ID ~0u

struct TriangleID {
  uint transform_index;
  uint primitive_index;
  uint triangle_index;
};

uint pack_triangle_id(uint transform_index, uint primitive_index, uint triangle_index) {
  uint res = 0;
  res |= triangle_index & TRIANGLE_MASK;
  res |= (primitive_index << TRIANGLE_BITS) & PRIMITIVE_MASK;
  res |= (transform_index << (TRIANGLE_BITS + PRIMITIVE_BITS)) & TRANSFORM_MASK;
  return res;
}

uint set_triangle_id(uint packed_id, uint triangle_index) {
  packed_id = packed_id ^ (packed_id & TRIANGLE_MASK);
  packed_id |= triangle_index & TRIANGLE_MASK;
  return packed_id;
}

uint pack_triangle_id(TriangleID tid) {
  return pack_triangle_id(tid.transform_index, tid.primitive_index, tid.triangle_index);
}

TriangleID unpack_triangle_id(uint tid) {
  TriangleID res = TriangleID(0, 0, 0);
  res.triangle_index = tid & TRIANGLE_MASK;
  res.primitive_index = (tid & PRIMITIVE_MASK) >> TRIANGLE_BITS;
  res.transform_index = (tid & TRANSFORM_MASK) >> (TRIANGLE_BITS + PRIMITIVE_BITS);
  return res;
}

vec3 barycentric_coords(vec3 P, vec3 a, vec3 b, vec3 c) {
  vec3 AB = b - a;
  vec3 AC = c - a;

  vec3 N = cross(AB, AC);
  float coef = 1.f/dot(N, N);
  
  vec3 BC = c - b;
  vec3 BP = P - b;
  float u = coef * dot(cross(BC, BP), N); 
  
  vec3 CA = a - c;
  vec3 CP = P - c;
  float v = coef * dot(cross(CA, CP), N);

  return vec3(u, v, 1 - u - v);
}

vec3 trace_barycentric_coords(vec3 V, vec3 a, vec3 b, vec3 c) {
  vec3 AB = b - a;
  vec3 AC = c - a;

  vec3 N = cross(AB, AC);

  float denom = dot(V, N);
  if (abs(denom) < 1e-6)
    return vec3(0, 0, 0);
  
  float t = dot(a, N)/denom;
  vec3 P = t * V;

  float coef = 1.f/dot(N, N);
  
  vec3 BC = c - b;
  vec3 BP = P - b;
  float u = coef * dot(cross(BC, BP), N); 
  
  vec3 CA = a - c;
  vec3 CP = P - c;
  float v = coef * dot(cross(CA, CP), N);

  return vec3(u, v, 1 - u - v);
}

struct Primitive {
  uint vertex_offset;
  uint index_offset;
  uint index_count;
  uint material_index;
};

struct Vertex {
  float pos_x, pos_y, pos_z;
  float norm_x, norm_y, norm_z;
  float u, v;
};

vec3 get_vertex_pos(in Vertex v) {
  return vec3(v.pos_x, v.pos_y, v.pos_z);
}

vec3 get_vertex_norm(in Vertex v) {
  return normalize(vec3(v.norm_x, v.norm_y, v.norm_z));
}

vec2 get_vertex_uv(in Vertex v) {
  return vec2(v.u, v.v);
}

struct Transform {
  mat4 model;
  mat4 normal;
};

struct Material {
  uint albedo_tex_index;
  uint metalic_roughness_index;
  uint flags;
  float alpha_cutoff;
};

uint lowbias32(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

#endif