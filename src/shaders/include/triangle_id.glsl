#ifndef TRIANGLE_ID_GLSL_INCLUDED
#define TRIANGLE_ID_GLSL_INCLUDED

#define TRIANGLE_BITS 20
#define DRAWCALL_BITS 12

#define TRIANGLE_MASK 0x000fffff
#define DRAWCALL_MASK 0xfff00000 

#define INVALID_TRIANGLE_ID ~0u

struct TriangleID {
  uint drawcall_index;
  uint triangle_index;
};

uint pack_triangle_id(uint drawcall_index, uint triangle_index) {
  uint res = 0;
  res |= triangle_index & TRIANGLE_MASK;
  res |= (drawcall_index << TRIANGLE_BITS) & DRAWCALL_MASK;
  return res;
}

/*uint set_triangle_id(uint packed_id, uint triangle_index) {
  packed_id = packed_id ^ (packed_id & TRIANGLE_MASK);
  packed_id |= triangle_index & TRIANGLE_MASK;
  return packed_id;
}*/

uint pack_triangle_id(TriangleID tid) {
  return pack_triangle_id(tid.drawcall_index, tid.triangle_index);
}

TriangleID unpack_triangle_id(uint tid) {
  TriangleID res = TriangleID(0, 0);
  res.triangle_index = tid & TRIANGLE_MASK;
  res.drawcall_index = (tid >> TRIANGLE_BITS) & ((1 << DRAWCALL_BITS) - 1);
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

struct Drawcall {
  uint transform_index;
  uint primitive_index;
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