
vec2 verts[] = vec2[](
  vec2(-1, -1),
  vec2(-1, 3),
  vec2(3, -1)
);

vec2 uv_coords[] = vec2[](
  vec2(0, 0),
  vec2(0, 2),
  vec2(2, 0)
);

#define OUT_VERTEX vec4(verts[gl_VertexIndex], 0, 1)
#define OUT_UV (uv_coords[gl_VertexIndex])