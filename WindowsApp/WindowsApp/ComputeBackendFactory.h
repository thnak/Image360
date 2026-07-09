#pragma once

#include "IComputeBackend.h"
#include "IImageCodec.h"
#include <memory>
#include <string>

namespace WindowsApp
{
    struct ComputeBackendSelection
    {
        std::shared_ptr<::WindowsApp::Compute::IComputeBackend> backend;
        std::shared_ptr<::WindowsApp::Compute::IImageCodec> codec;
        bool usedGpu = false;
        // TaskScheduler's in-flight task window - GPU tasks each hold a
        // full-resolution VRAM buffer (kept low, see TaskScheduler.h);
        // CPU-backend kernels are single-threaded internally, so all
        // parallelism comes from this window instead (core-count-derived).
        size_t recommendedMaxInFlight = 2;
        // Project chunk size - GPU: VRAM-derived (ProjectManager::
        // RecommendedChunkSize); CPU: system-RAM-derived
        // (ProjectManager::RecommendedChunkSizeForCpu).
        int recommendedChunkSize = 1024;
        std::wstring statusMessage;
    };

    // Tries CudaPipeline+NvJpegCodec first; falls back to
    // CpuComputeBackend+JpegCodec if CUDA/GPU init fails (no compatible
    // GPU, driver/toolkit missing, etc.) - this is what makes the app
    // usable on machines with no GPU at all, replacing the previous
    // behavior of aborting the whole stitch run on CUDA init failure.
    ComputeBackendSelection SelectComputeBackend();
}
