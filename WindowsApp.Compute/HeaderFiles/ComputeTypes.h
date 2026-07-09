#pragma once

#include <cstdint>
#include <cstddef>

namespace WindowsApp { namespace Compute
{
    // Plain C++ types (no CUDA headers), shared by IComputeBackend/
    // IImageCodec, CudaPipeline, CpuComputeBackend, and plain-MSVC-compiled
    // callers (WindowsApp.Core, WindowsApp.Tests) alike. Split out from
    // CudaPipeline.h so code that only needs the data types doesn't have to
    // include either backend's interface header.

    // Compute device information - populated with GPU specifics by
    // CudaPipeline, or with CPU-equivalent cosmetic values (deviceId = -1,
    // name = "CPU (AVX2)" etc., totalMemory/freeMemory = system RAM,
    // hasTensorCores = false) by CpuComputeBackend.
    struct GpuInfo
    {
        int deviceId = -1;
        char name[256] = {};
        size_t totalMemory = 0;
        size_t freeMemory = 0;
        int computeMajor = 0;
        int computeMinor = 0;
        int maxThreadsPerBlock = 0;
        int multiProcessorCount = 0;
        bool hasTensorCores = false;  // Volta (SM 7.0+) or Ampere (SM 8.0+)
    };

    // Result codes
    enum class ComputeResult
    {
        SUCCESS = 0,
        NO_GPU = 1,
        INVALID_PARAM = 2,
        OUT_OF_MEMORY = 3,
        KERNEL_LAUNCH_FAILED = 4,
        CUDA_ERROR = 5
    };

    struct FeaturePoint
    {
        float x = 0.0f;
        float y = 0.0f;
    };

    using BriefDescriptor = uint64_t[4]; // 256-bit binary descriptor

    struct MatchResult
    {
        int indexA = 0;
        int indexB = 0;
        int hammingDistance = 0;
    };
}}
