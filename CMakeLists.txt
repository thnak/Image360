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

# Off by default: needs the real-photo fixtures and is meant to run on
# the actual self-hosted Windows CI runner (which sets this ON
# explicitly), not as a routine local/dev-loop build - see
# tests/pipeline_e2e_real_photos/main.cpp.
option(IMAGE360_ENABLE_REAL_PHOTO_TESTS "Build/register the real-photo pipeline e2e test (CI-only)" OFF)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Image360Engine: CUDA compiler found (${CMAKE_CUDA_COMPILER}) - building WindowsApp.Compute")
    set(IMAGE360_HAVE_CUDA ON)
else()
    message(STATUS "Image360Engine: no CUDA compiler found - skipping WindowsApp.Compute (GPU features unavailable in this build)")
    set(IMAGE360_HAVE_CUDA OFF)
endif()

# Vulkan compute backend (WindowsApp.Vulkan) - independent of CUDA, gives
# non-NVIDIA GPUs (e.g. the win-amd CI runner's AMD card) a real GPU path
# instead of falling back straight to CPU. Needs both the Vulkan SDK
# (headers/loader) and glslc (GLSL->SPIR-V compiler, normally bundled with
# the SDK) - if either is missing, skip it the same way CUDA is skipped
# without nvcc, rather than hard-failing the whole configure.
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    find_program(GLSLC_EXECUTABLE
        NAMES glslc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin32"
    )
    if(GLSLC_EXECUTABLE)
        message(STATUS "Image360Engine: Vulkan SDK found (${Vulkan_INCLUDE_DIRS}) and glslc found (${GLSLC_EXECUTABLE}) - building WindowsApp.Vulkan")
        set(IMAGE360_HAVE_VULKAN ON)
    else()
        message(STATUS "Image360Engine: Vulkan SDK found but glslc not found - skipping WindowsApp.Vulkan")
        set(IMAGE360_HAVE_VULKAN OFF)
    endif()
else()
    message(STATUS "Image360Engine: no Vulkan SDK found - skipping WindowsApp.Vulkan (GPU features unavailable in this build)")
    set(IMAGE360_HAVE_VULKAN OFF)
endif()

add_subdirectory(WindowsApp.Core)

if(IMAGE360_HAVE_CUDA)
    enable_language(CUDA)
    add_subdirectory(WindowsApp.Compute)
endif()

if(IMAGE360_HAVE_VULKAN)
    add_subdirectory(WindowsApp.Vulkan)
endif()

enable_testing()
add_subdirectory(tests/engine_smoke)
add_subdirectory(tests/raw_probe)
add_subdirectory(tests/render_blend)
add_subdirectory(tests/pipeline_e2e)

if(IMAGE360_HAVE_VULKAN)
    add_subdirectory(tests/vulkan_smoke)
    add_subdirectory(tests/pipeline_e2e_vulkan)
endif()

if(IMAGE360_ENABLE_REAL_PHOTO_TESTS)
    add_subdirectory(tests/pipeline_e2e_real_photos)
endif()
