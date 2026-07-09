# WindowsApp.Compute: CUDA kernels (warp/blend/median-stack, tensor-core
# homography + bundle-adjustment normal equations, nvJPEG codec). Only
# added to the build when the top-level CMakeLists.txt found a CUDA
# compiler - see the CMAKE_CUDA_COMPILER check there.

find_package(CUDAToolkit REQUIRED)

set(COMPUTE_SOURCES
    SourceFiles/CudaPipeline.cpp
    SourceFiles/NvJpegCodec.cpp
    SourceFiles/median_stack.cu
    SourceFiles/tensor_ops.cu
    SourceFiles/demosaic.cu
    SourceFiles/features.cu
    SourceFiles/color_transfer.cu
)

# CudaPipeline.cpp/NvJpegCodec.cpp are compiled by nvcc in the MSBuild
# project too (CudaCompile, not ClCompile) since they call CUDA runtime
# APIs directly - mirror that here.
set_source_files_properties(
    SourceFiles/CudaPipeline.cpp
    SourceFiles/NvJpegCodec.cpp
    PROPERTIES LANGUAGE CUDA
)

add_library(windowsapp_compute SHARED ${COMPUTE_SOURCES})

target_include_directories(windowsapp_compute
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/HeaderFiles
)

target_compile_definitions(windowsapp_compute PRIVATE WINDOWSAPPCOMPUTE_EXPORTS)

# Matches the compute_89,sm_89 (Ada Lovelace/RTX 40-series) pin in the
# MSBuild project - override with -DIMAGE360_CUDA_ARCHITECTURES=... for
# other GPUs.
set(IMAGE360_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures to build WindowsApp.Compute for")
set_target_properties(windowsapp_compute PROPERTIES CUDA_ARCHITECTURES "${IMAGE360_CUDA_ARCHITECTURES}")

target_link_libraries(windowsapp_compute PRIVATE CUDA::nvjpeg CUDA::cudart)
