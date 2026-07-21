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
        CUDA_ERROR = 5,
        // A real, typed "not implemented on this backend yet" signal - for
        // ops where only a subset of IComputeBackend's implementations
        // have a real kernel (e.g. BlockMatchAlign/RobustMergeAccumulate,
        // CPU-only as of docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md).
        // Deliberately distinct from every other failure code so a caller
        // (or a test) can tell "this backend doesn't support this op" apart
        // from "the op was attempted and failed" - see
        // docs/COMPUTATIONAL_PHOTOGRAPHY.md SS4's backend-coverage-matrix
        // guidance, this is that tracking mechanism's first real use.
        NOT_SUPPORTED = 6
    };

    struct FeaturePoint
    {
        float x = 0.0f;
        float y = 0.0f;
    };

    // Per-tile local translation (BlockMatchAlign's output, docs/
    // COMPUTATIONAL_PHOTOGRAPHY.md SS3) - deliberately not a Homography:
    // this is a dense per-tile field, not one global transform, so it
    // needs its own tiny type rather than reusing the panorama path's
    // Homography for a one-component (translation-only) special case.
    struct TileOffset
    {
        int dx = 0;
        int dy = 0;
    };

    using BriefDescriptor = uint64_t[4]; // 256-bit binary descriptor

    struct MatchResult
    {
        int indexA = 0;
        int indexB = 0;
        int hammingDistance = 0;
    };
}}
