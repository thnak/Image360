#pragma once

#include "IComputeBackend.h"
#include "IImageCodec.h"
#include <memory>
#include <string>

namespace WindowsApp
{
    // User-facing compute backend choice (e.g. a settings ComboBox) -
    // Auto reproduces the original CUDA-then-CPU fallback behavior with
    // Vulkan added as the middle tier; the other 3 values force that one
    // specific backend (still with a CPU fallback if it fails to
    // initialize, since a hard failure here would abort the whole app
    // rather than just under-deliver on GPU acceleration).
    enum class ComputeBackendKind
    {
        Auto,
        Cuda,
        Vulkan,
        Cpu,
    };

    struct ComputeBackendSelection
    {
        std::shared_ptr<::WindowsApp::Compute::IComputeBackend> backend;
        std::shared_ptr<::WindowsApp::Compute::IImageCodec> codec;
        bool usedGpu = false;
        // Which backend actually got selected (may differ from the
        // requested ComputeBackendKind if that one failed to initialize
        // and this fell back to CPU) - lets a settings UI reflect back
        // what's really running, not just what was asked for.
        ComputeBackendKind kind = ComputeBackendKind::Cpu;
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

    // preferred == Auto tries CudaPipeline+NvJpegCodec, then
    // VulkanPipeline+JpegCodec, then CpuComputeBackend+JpegCodec - the
    // first one that initializes successfully wins. Cuda/Vulkan/Cpu force
    // that one specific backend, still falling back to CPU if the forced
    // choice fails to initialize (no compatible GPU, driver/toolkit
    // missing, etc.) rather than aborting the whole stitch run - this is
    // what makes the app usable on any machine regardless of what's
    // installed.
    ComputeBackendSelection SelectComputeBackend(ComputeBackendKind preferred = ComputeBackendKind::Auto);
}
