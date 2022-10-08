cmake_minimum_required(VERSION 3.0)

project(cgtask3)

find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g -pthread")

include_directories(.)

set(spv-dir "lib/spirv-cross")
set(imgui-dir "lib/imgui")

add_subdirectory(gpu)

add_executable(main 
  rendergraph/resources.cpp
  rendergraph/rendergraph.cpp
  rendergraph/gpu_ctx.cpp

  main.cpp
  backbuffer_subpass2.cpp
  util_passes.cpp
  scene_renderer.cpp
  gpu_transfer.cpp
  defered_shading.cpp
  image_readback.cpp
  gtao.cpp
  imgui_pass.cpp
  downsample_pass.cpp
  trace_samples.cpp
  ssr.cpp
  advanced_ssr.cpp
  taa.cpp
  
  scene/scene.cpp
  scene/scene_as.cpp
  scene/images.cpp)

target_link_libraries(main vk-gpu ${SDL2_LIBRARIES} ${Vulkan_LIBRARIES})