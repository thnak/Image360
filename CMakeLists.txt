# Cross-platform build for the WindowsApp.Core / WindowsApp.Compute engine.
#
# This is additive to, and independent of, the Windows-only MSBuild solution
# (WindowsApp.slnx) that builds the WinUI shell + packaging - that remains
# the authoritative Windows build. This CMake build exists so the engine
# (orchestration + GPU kernels) can be compiled and smoke-tested headlessly
# on any platform with a C++20 compiler, and with CUDA when available.

cmake_minimum_required(VERSION 3.20)
project(Image360Engine LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Image360Engine: CUDA compiler found (${CMAKE_CUDA_COMPILER}) - building WindowsApp.Compute")
    set(IMAGE360_HAVE_CUDA ON)
else()
    message(STATUS "Image360Engine: no CUDA compiler found - skipping WindowsApp.Compute (GPU features unavailable in this build)")
    set(IMAGE360_HAVE_CUDA OFF)
endif()

add_subdirectory(WindowsApp.Core)

if(IMAGE360_HAVE_CUDA)
    enable_language(CUDA)
    add_subdirectory(WindowsApp.Compute)
endif()

enable_testing()
add_subdirectory(tests/engine_smoke)
add_subdirectory(tests/raw_probe)
add_subdirectory(tests/pipeline_e2e)
