#ifndef COMPRESSED_PLANE_GLSL_INCLUDED
#define COMPRESSED_PLANE_GLSL_INCLUDED

struct VkAABB {
  float min_x;
  float min_y;
  float min_z;
  float max_x;
  float max_y;
  float max_z;
};

struct CompressedPlane {
  uint packed_normal;
  uint pos_x;
  uint pos_y;
  uint size_depth;
};

#endif