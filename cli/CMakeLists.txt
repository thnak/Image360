# image360-cli: headless CLI front end (docs/COMPUTATIONAL_PHOTOGRAPHY.md
# SS6, docs/superpowers/plans/2026-07-22-cli-front-end.md). CMake-only for
# now (no WindowsApp.slnx integration - see the plan doc's Tech stack
# section). Links windowsapp_vulkan when available (IMAGE360_HAVE_VULKAN,
# same gate tests/pipeline_e2e_vulkan already uses) - CUDA is a tracked
# follow-up (plan doc scope decision 3), not linked here yet.

add_executable(image360_cli main.cpp)
target_link_libraries(image360_cli PRIVATE windowsapp_core)

if(IMAGE360_HAVE_VULKAN)
    target_link_libraries(image360_cli PRIVATE windowsapp_vulkan)
    target_compile_definitions(image360_cli PRIVATE IMAGE360_CLI_HAVE_VULKAN)
endif()
