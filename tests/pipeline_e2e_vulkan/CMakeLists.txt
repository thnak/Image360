# Full end-to-end pipeline test driven by VulkanPipeline instead of
# CpuComputeBackend - reuses tests/pipeline_e2e's fixtures and shared
# scenario code (PipelineE2ETestSupport.h). Only built when
# IMAGE360_HAVE_VULKAN is set (see root CMakeLists.txt). Skips gracefully
# (exit 0) on machines with no usable Vulkan device, same as
# tests/vulkan_smoke.

add_executable(pipeline_e2e_vulkan_tests main.cpp)
target_include_directories(pipeline_e2e_vulkan_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../pipeline_e2e"
)
target_link_libraries(pipeline_e2e_vulkan_tests PRIVATE windowsapp_core windowsapp_vulkan)
target_compile_definitions(pipeline_e2e_vulkan_tests PRIVATE
    PIPELINE_E2E_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../pipeline_e2e/fixtures"
)

add_test(NAME pipeline_e2e_vulkan_tests COMMAND pipeline_e2e_vulkan_tests)
