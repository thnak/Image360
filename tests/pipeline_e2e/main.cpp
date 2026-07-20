// Full pipeline integration test (CPU compute backend). See
// PipelineE2ETestSupport.h for the shared scenario bodies - this file only
// wires up CpuComputeBackend and drives the shared test entry point. The
// Vulkan-backed sibling of this test (tests/pipeline_e2e_vulkan/main.cpp)
// reuses the exact same fixtures and scenarios against VulkanPipeline
// instead, to prove GPU and CPU compute backends produce equivalent
// pipeline results.

#include "PipelineE2ETestSupport.h"
#include "HeaderFiles/CpuComputeBackend.h"

#include <filesystem>
#include <iostream>
#include <memory>

int main()
{
    namespace fs = std::filesystem;
    using namespace WindowsApp::Core;
    using namespace WindowsApp::Compute;

    fs::path fixturesDir(PIPELINE_E2E_FIXTURES_DIR);

    auto computeBackend = std::make_shared<CpuComputeBackend>();
    if (computeBackend->Initialize() != ComputeResult::SUCCESS)
    {
        std::cerr << "FAIL: CpuComputeBackend::Initialize" << std::endl;
        return 1;
    }

    int result = WindowsApp::Testing::RunPipelineE2ETests(fixturesDir, computeBackend);
    computeBackend->Shutdown();
    return result;
}
