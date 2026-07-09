# Full end-to-end pipeline test: loads synthetic RAW (DNG) tiles with known
# overlap, drives PipelineDriver through all 4 stages on the CPU backend,
# and checks the recovered geometry against the known ground truth.

add_executable(pipeline_e2e_tests main.cpp)
target_link_libraries(pipeline_e2e_tests PRIVATE windowsapp_core)
target_compile_definitions(pipeline_e2e_tests PRIVATE
    PIPELINE_E2E_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

add_test(NAME pipeline_e2e_tests COMMAND pipeline_e2e_tests)
