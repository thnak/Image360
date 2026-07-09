# Manual diagnostic tool (not registered with ctest) for validating
# hand-rolled/synthetic RAW files against the vendored LibRaw via the same
# ImageLoader calls RawIngestExecutor/AlignExecutor make.

add_executable(raw_probe main.cpp)
target_link_libraries(raw_probe PRIVATE windowsapp_core)
