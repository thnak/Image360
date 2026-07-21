# MFNR end-to-end test (docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md
# Task 6): kernel-level BlockMatchAlign/RobustMergeAccumulate checks plus a
# full burst pipeline run on the CPU backend, gated on PSNR vs. a clean
# synthetic scene. No committed fixtures - frames are generated and JPEG-
# encoded in-process.

add_executable(pipeline_e2e_burst_tests main.cpp)
target_link_libraries(pipeline_e2e_burst_tests PRIVATE windowsapp_core)

add_test(NAME pipeline_e2e_burst_tests COMMAND pipeline_e2e_burst_tests)
