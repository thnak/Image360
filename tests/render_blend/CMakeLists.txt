# Direct unit tests for Kernels::SeamBlend (WindowsApp.Core) - synthetic
# in-memory buffers, no image fixtures needed. Mirrors tests/engine_smoke's
# exact pattern.

add_executable(render_blend_tests main.cpp)
target_link_libraries(render_blend_tests PRIVATE windowsapp_core)

add_test(NAME render_blend_tests COMMAND render_blend_tests)
