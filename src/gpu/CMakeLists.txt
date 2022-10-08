cmake_minimum_required(VERSION 3.0)

project(vk-gpu)
find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g -pthread")

set(spv-dir "../lib/spirv-cross")
set(imgui-dir "../lib/imgui")
set(lib-dir "../lib")

set(spirv-cross-sources
	${spv-dir}/spirv_cross.cpp
	${spv-dir}/spirv_parser.cpp
	${spv-dir}/spirv_cross_parsed_ir.cpp
	${spv-dir}/spirv_cfg.cpp
  ${spv-dir}/spirv_glsl.cpp)

set(imgui-sources
  ${imgui-dir}/imgui_draw.cpp
  ${imgui-dir}/imgui_impl_sdl.cpp
  ${imgui-dir}/imgui_impl_vulkan.cpp
  ${imgui-dir}/imgui_tables.cpp
  ${imgui-dir}/imgui_widgets.cpp
  ${imgui-dir}/imgui.cpp)

add_library(vk-gpu STATIC
  common.cpp
  driver.cpp
  swapchain.cpp
  pipelines.cpp
  resources.cpp
  managed_resources.cpp
  descriptors.cpp
  shader_program.cpp
  shader.cpp
  cmd_buffers.cpp
  samplers.cpp
  framebuffers.cpp
  gpu.cpp
  #"${lib-dir}/spirv-reflect/spirv-reflect.cpp"
  ../lib/spirv-reflect/spirv_reflect.cpp
  ${imgui-sources}
  ../lib/volk.c)

target_link_libraries(vk-gpu dl ${Vulkan_LIBRARIES})
