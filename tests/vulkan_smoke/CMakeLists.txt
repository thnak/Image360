# Vulkan compute backend smoke test - only built when IMAGE360_HAVE_VULKAN
# is set (see root CMakeLists.txt). Gracefully skips its GPU checks (rather
# than failing) on a machine with the SDK but no usable Vulkan device/ICD -
# this backend is optional infrastructure, same fallback philosophy as the
# CUDA backend.

add_executable(vulkan_smoke_tests main.cpp)
target_link_libraries(vulkan_smoke_tests PRIVATE windowsapp_core windowsapp_vulkan)

add_test(NAME vulkan_smoke_tests COMMAND vulkan_smoke_tests)
