# Headless smoke test for windowsapp_core - proves ProjectManager (SQLite)
# and StorageEngine (chunked file I/O) round-trip correctly on this
# platform without any Windows/WinUI/MSTest dependency.

add_executable(engine_smoke_tests main.cpp)
target_link_libraries(engine_smoke_tests PRIVATE windowsapp_core)

add_test(NAME engine_smoke_tests COMMAND engine_smoke_tests)
