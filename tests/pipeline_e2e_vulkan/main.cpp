// Full pipeline integration test (Vulkan compute backend). Reuses the exact
// same scenarios and fixtures as tests/pipeline_e2e/main.cpp (see
// PipelineE2ETestSupport.h) but drives Ingest/Align/Optimize/Render through
// VulkanPipeline instead of CpuComputeBackend - this is the whole-engine
// proof that the GPU compute path (WarpPerspective/MedianStack/ApplyGain/
// DemosaicBayer, see WindowsApp.Vulkan/shaders/) produces a working
// panorama, not just that its kernels match the CPU reference in isolation
// (that per-kernel check already lives in tests/vulkan_smoke).
//
// If no usable Vulkan device is present on this machine, this is a skip,
// not a failure - same optional-infrastructure philosophy as
// tests/vulkan_smoke and the CUDA backend's own fallback.

#include "PipelineE2ETestSupport.h"
#include "HeaderFiles/VulkanPipeline.h"

#include <filesystem>
#include <iostream>
#include <memory>

int main()
{
    namespace fs = std::filesystem;
    using namespace WindowsApp::Compute;

    fs::path fixturesDir(PIPELINE_E2E_FIXTURES_DIR);

    auto computeBackend = std::make_shared<VulkanPipeline>();
    if (computeBackend->Initialize() != ComputeResult::SUCCESS)
    {
        std::cout << "SKIP: no usable Vulkan device on this machine (" << computeBackend->GetLastError() << ")" << std::endl;
        std::cout << "pipeline_e2e_vulkan_tests: 0 failures (skipped - no Vulkan device)" << std::endl;
        return 0;
    }

    GpuInfo info = computeBackend->GetGpuInfo();
    std::cout << "Vulkan device: " << info.name << std::endl;

    int result = WindowsApp::Testing::RunPipelineE2ETests(fixturesDir, computeBackend);
    computeBackend->Shutdown();
    return result;
}
