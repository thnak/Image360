// Vulkan compute backend smoke test: constructs VulkanPipeline, and if a
// usable Vulkan device is actually present, runs WarpPerspective/
// MedianStack/ApplyGain/DemosaicBayer against synthetic data and diffs
// the results against WindowsApp::Core::Kernels::Scalar::* - the same
// reference kernels CpuComputeBackend uses (see
// tests/engine_smoke/main.cpp's RunCrossTierSimdKernelChecks for the
// established pattern this mirrors). If Initialize() fails (no ICD/driver
// on this machine), that's logged and treated as a skip, not a failure -
// this backend is optional infrastructure, same fallback philosophy as
// the CUDA backend.

#include "HeaderFiles/VulkanPipeline.h"
#include "HeaderFiles/WarpPerspectiveKernels.h"
#include "HeaderFiles/MedianStackKernels.h"
#include "HeaderFiles/BayerDemosaicKernels.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace WindowsApp::Core;
using namespace WindowsApp::Compute;

namespace
{
    int g_failures = 0;

    void Check(bool condition, const char* what)
    {
        if (!condition)
        {
            std::cerr << "FAIL: " << what << std::endl;
            ++g_failures;
        }
        else
        {
            std::cout << "OK: " << what << std::endl;
        }
    }

    int MaxAbsDiff(const std::vector<unsigned short>& a, const std::vector<unsigned short>& b)
    {
        int maxDiff = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
            if (diff > maxDiff) maxDiff = diff;
        }
        return maxDiff;
    }

    void CheckApplyGain(VulkanPipeline& gpu)
    {
        std::vector<unsigned short> data = { 100, 200, 300, 60000 };
        std::vector<unsigned short> expected = data;

        for (auto& v : expected)
        {
            float val = static_cast<float>(v) * 1.5f;
            v = static_cast<unsigned short>((std::min)((std::max)(val, 0.0f), 65535.0f));
        }

        Check(gpu.ApplyGain(data.data(), static_cast<int>(data.size()), 1.5f) == ComputeResult::SUCCESS,
              "VulkanPipeline::ApplyGain succeeds");
        Check(data == expected, "VulkanPipeline::ApplyGain matches scalar reference exactly");
    }

    void CheckWarpPerspective(VulkanPipeline& gpu)
    {
        const int w = 40, h = 30;
        const size_t planeSize = static_cast<size_t>(w) * h * 3;
        std::vector<unsigned short> src(planeSize);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                    src[(static_cast<size_t>(y) * w + x) * 3 + c] =
                        static_cast<unsigned short>((x * 37 + y * 53 + c * 17) % 4001);

        float angle = 0.06f;
        float ca = std::cos(angle), sa = std::sin(angle);
        float homography[9] = { ca, -sa, 1.7f, sa, ca, 0.9f, 0.0f, 0.0f, 1.0f };

        std::vector<unsigned short> outScalar(planeSize, 0);
        Kernels::Scalar::WarpPerspective(src.data(), w, h, outScalar.data(), w, h, homography, 0, 0);

        std::vector<unsigned short> outGpu(planeSize, 0);
        Check(gpu.WarpPerspective(src.data(), w, h, outGpu.data(), w, h, homography, 0, 0) == ComputeResult::SUCCESS,
              "VulkanPipeline::WarpPerspective succeeds");
        // Bilinear blend is floating-point - allow a couple of ULPs of
        // cross-implementation rounding noise, same tolerance
        // tests/engine_smoke/main.cpp applies across CPU SIMD tiers.
        Check(MaxAbsDiff(outScalar, outGpu) <= 2, "VulkanPipeline::WarpPerspective within tolerance of scalar reference");
    }

    void CheckMedianStack(VulkanPipeline& gpu)
    {
        const int width = 17, height = 13;
        const int numInputs = 21;
        const size_t planeSize = static_cast<size_t>(width) * height * 3;

        std::mt19937 rng(777);
        std::uniform_int_distribution<int> pixelDist(0, 4000);
        std::vector<std::vector<unsigned short>> inputBuffers(numInputs);
        std::vector<const unsigned short*> inputPtrs(numInputs);
        for (int i = 0; i < numInputs; ++i)
        {
            inputBuffers[i].resize(planeSize);
            for (auto& v : inputBuffers[i]) v = static_cast<unsigned short>(pixelDist(rng));
            inputPtrs[i] = inputBuffers[i].data();
        }
        for (int i = 0; i < 3; ++i) inputBuffers[i][100] = 60000;

        std::vector<unsigned short> outScalar(planeSize, 0);
        Kernels::Scalar::MedianStack(inputPtrs.data(), numInputs, outScalar.data(), width, height, 2.0f);

        std::vector<unsigned short> outGpu(planeSize, 0);
        Check(gpu.MedianStack(inputPtrs.data(), numInputs, outGpu.data(), width, height, 2.0f) == ComputeResult::SUCCESS,
              "VulkanPipeline::MedianStack succeeds");
        Check(outGpu == outScalar, "VulkanPipeline::MedianStack bit-exact vs scalar reference");
    }

    void CheckDemosaicBayer(VulkanPipeline& gpu)
    {
        const int w = 24, h = 20;
        const uint32_t filters = 0x94949494; // LibRaw's standard RGGB encoding
        std::vector<unsigned short> cfa(static_cast<size_t>(w) * h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                cfa[static_cast<size_t>(y) * w + x] = static_cast<unsigned short>(200 + (x * 31 + y * 19) % 3000);

        const unsigned short blackLevel = 64;
        const float camMul[4] = { 1.8f, 1.0f, 1.5f, 1.0f };
        const float rgbCam[3][4] = {
            { 1.9f, -0.7f, -0.2f, 0.0f },
            { -0.3f, 1.5f, -0.2f, 0.0f },
            { 0.0f, -0.4f, 1.6f, 0.0f },
        };

        std::vector<unsigned short> outScalar(static_cast<size_t>(w) * h * 3, 0);
        Kernels::Scalar::DemosaicBayer(cfa.data(), w, h, blackLevel, camMul, rgbCam, filters, outScalar.data());

        std::vector<unsigned short> outGpu(static_cast<size_t>(w) * h * 3, 0);
        Check(gpu.DemosaicBayer(cfa.data(), w, h, blackLevel, camMul, rgbCam, filters, outGpu.data()) == ComputeResult::SUCCESS,
              "VulkanPipeline::DemosaicBayer succeeds");

        int diff = MaxAbsDiff(outScalar, outGpu);
        int mismatches = 0;
        for (size_t i = 0; i < outScalar.size(); ++i)
        {
            if (outScalar[i] != outGpu[i]) ++mismatches;
        }
        std::cout << "DemosaicBayer diff: max=" << diff << " mismatched=" << mismatches
                   << "/" << outScalar.size() << std::endl;

        // Real GPU hardware isn't guaranteed bit-identical to MSVC's CPU
        // scalar path for floating-point pixel math (FMA contraction etc.)
        // - same documented tolerance class as WarpPerspective's bilinear
        // blend above, not a functional bug. Verified bit-exact against
        // Mesa llvmpipe's software rasterizer; only real hardware needs
        // this slack.
        Check(diff <= 1, "VulkanPipeline::DemosaicBayer within tolerance of scalar reference");
    }
}

int main()
{
    VulkanPipeline gpu;
    ComputeResult initResult = gpu.Initialize();

    if (initResult != ComputeResult::SUCCESS)
    {
        std::cout << "SKIP: no usable Vulkan device on this machine (" << gpu.GetLastError() << ")" << std::endl;
        std::cout << "vulkan_smoke_tests: 0 failures (skipped - no Vulkan device)" << std::endl;
        return 0;
    }

    GpuInfo info = gpu.GetGpuInfo();
    std::cout << "Vulkan device: " << info.name << std::endl;

    CheckApplyGain(gpu);
    CheckWarpPerspective(gpu);
    CheckMedianStack(gpu);
    CheckDemosaicBayer(gpu);

    gpu.Shutdown();

    std::cout << "vulkan_smoke_tests: " << g_failures << " failures" << std::endl;
    return g_failures == 0 ? 0 : 1;
}
