# Full pipeline e2e test using real phone-camera JPEGs (not synthetic
# fixtures) - see main.cpp's header comment for why this exists and what
# it does/doesn't check. Only built/registered when
# IMAGE360_ENABLE_REAL_PHOTO_TESTS is ON (see root CMakeLists.txt) - the
# self-hosted Windows CI workflow turns this on explicitly; a default
# local build does not build or run it.

add_executable(pipeline_e2e_real_photos_tests main.cpp)
target_link_libraries(pipeline_e2e_real_photos_tests PRIVATE windowsapp_core)
target_compile_definitions(pipeline_e2e_real_photos_tests PRIVATE
    REAL_PHOTOS_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

add_test(NAME pipeline_e2e_real_photos_tests COMMAND pipeline_e2e_real_photos_tests)
