#ifndef TRACE_SAMPLES_GLSL_INCLUDED
#define TRACE_SAMPLES_GLSL_INCLUDED

#ifndef USE_SAMPLES_TRACE
#define USE_SAMPLES_TRACE 0
#endif

#ifndef HEATMAP_SET
#define HEATMAP_SET 0
#endif

#ifndef HEATMAP_BINDING
#define HEATMAP_BINDING 7
#endif

#ifndef START_X
#define START_X 0
#endif

#ifndef END_X
#define END_X 1
#endif

#ifndef START_Y
#define START_Y 0
#endif

#ifndef END_Y
#define END_Y 1
#endif

#if USE_SAMPLES_TRACE

layout(set = HEATMAP_SET, binding = HEATMAP_BINDING, r32ui) uniform uimage2D samples_heatmap;

#define TRACE_SAMPLE_UV(uv, pos) if ((uv).x > (START_X) && (uv).x < (END_X) && (uv).y > (START_Y) && (uv).y < (END_Y)) \
  imageAtomicAdd(samples_heatmap, ivec2((pos).xy * imageSize(samples_heatmap).xy), 1)

#else
#define TRACE_SAMPLE_UV(uv, pos)
#endif

#endif