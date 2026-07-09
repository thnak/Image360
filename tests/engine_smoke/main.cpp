// Headless smoke test: proves windowsapp_core (ProjectManager + SQLite,
// StorageEngine + chunked file I/O) works end-to-end on this platform
// without any WinUI/MSTest dependency. Not a substitute for
// WindowsApp.Tests - just a build-time sanity check for the CMake path.

#include "HeaderFiles/ProjectManager.h"
#include "HeaderFiles/StorageEngine.h"
#include "HeaderFiles/TextEncoding.h"
#include "HeaderFiles/RansacHomography.h"
#include "HeaderFiles/BundleAdjustment.h"
#include "HeaderFiles/CpuComputeBackend.h"
#include "HeaderFiles/CpuSimdDetect.h"
#include "HeaderFiles/MedianStackKernels.h"
#include "HeaderFiles/WarpPerspectiveKernels.h"
#include "HeaderFiles/BayerDemosaicKernels.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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

    // GPU-independent since the CPU compute backend work replaced
    // CudaPipeline::TensorEstimateHomography/TensorSolveNormalEquations
    // with portable HomographyMath/LinearSolve - these run on any CPU.
    void RunRansacAndBundleAdjustmentChecks()
    {
        {
            std::mt19937 rng(123);
            std::uniform_real_distribution<float> coord(0.0f, 500.0f);
            std::vector<std::pair<FeaturePoint, FeaturePoint>> correspondences;
            for (int i = 0; i < 20; ++i)
            {
                FeaturePoint src{ coord(rng), coord(rng) };
                correspondences.emplace_back(src, FeaturePoint{ src.x + 5.0f, src.y + 7.0f });
            }

            RansacResult result = RunRansacHomography(correspondences, 200, 1.0f);
            Check(result.success && result.inlierCount == 20, "RunRansacHomography converges on a clean translation");
            Check(std::fabs(result.homography.h[2] - 5.0f) < 0.5f && std::fabs(result.homography.h[5] - 7.0f) < 0.5f,
                  "RunRansacHomography recovers the known translation");
        }

        {
            const float trueDx = 3.0f, trueDy = -2.0f;
            std::vector<int> nonReferenceImageIds = { 1 };
            std::vector<BaCorrespondence> correspondences;
            for (int i = 0; i < 10; ++i)
            {
                float x = static_cast<float>(i * 37 % 200);
                float y = static_cast<float>(i * 53 % 200);
                BaCorrespondence corr;
                corr.imageA = 0;
                corr.imageB = 1;
                corr.pointB = FeaturePoint{ x, y };
                corr.pointA = FeaturePoint{ x + trueDx, y + trueDy };
                correspondences.push_back(corr);
            }

            BaCheckpoint checkpoint;
            checkpoint.lambda = 1e-3f;
            checkpoint.parameters = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

            bool converged = false;
            for (int iter = 0; iter < 50 && !converged; ++iter)
            {
                LmStepResult step = RunOneLmIteration(nonReferenceImageIds, correspondences, checkpoint);
                checkpoint = step.checkpoint;
                converged = step.converged;
            }

            Check(converged, "RunOneLmIteration converges on a known translation");
            Check(std::fabs(checkpoint.parameters[2] - trueDx) < 0.1f && std::fabs(checkpoint.parameters[5] - trueDy) < 0.1f,
                  "RunOneLmIteration recovers the known translation");
        }
    }

    void RunCpuComputeBackendChecks()
    {
        CpuComputeBackend backend;
        Check(backend.Initialize() == ComputeResult::SUCCESS, "CpuComputeBackend::Initialize");
        Check(backend.IsInitialized(), "CpuComputeBackend::IsInitialized after Initialize");

        unsigned short gainData[4] = { 100, 200, 300, 400 };
        Check(backend.ApplyGain(gainData, 4, 2.0f) == ComputeResult::SUCCESS, "CpuComputeBackend::ApplyGain succeeds");
        Check(gainData[0] == 200 && gainData[1] == 400 && gainData[2] == 600 && gainData[3] == 800,
              "CpuComputeBackend::ApplyGain produces the expected values");

        const int w = 32, h = 32;
        std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3, 128);
        for (int y = 8; y < 16; ++y)
            for (int x = 8; x < 16; ++x)
                for (int c = 0; c < 3; ++c)
                    rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = 255;

        std::vector<FeaturePoint> points(100);
        std::vector<BriefDescriptor> descriptors(100);
        int featureCount = 0;
        Check(backend.DetectAndDescribeFeatures(rgb.data(), w, h, points.data(), descriptors.data(), &featureCount, 100)
                  == ComputeResult::SUCCESS,
              "CpuComputeBackend::DetectAndDescribeFeatures succeeds");
        Check(featureCount > 0, "CpuComputeBackend::DetectAndDescribeFeatures finds at least one corner");

        std::vector<unsigned short> srcImg(static_cast<size_t>(w) * h * 3, 1000);
        std::vector<unsigned short> dstImg(static_cast<size_t>(w) * h * 3, 0);
        float identityH[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        Check(backend.WarpPerspective(srcImg.data(), w, h, dstImg.data(), w, h, identityH, 0, 0) == ComputeResult::SUCCESS,
              "CpuComputeBackend::WarpPerspective succeeds");
        Check(dstImg[(static_cast<size_t>(h / 2) * w + w / 2) * 3] == 1000,
              "CpuComputeBackend::WarpPerspective with identity homography copies pixels through");

        const unsigned short* inputs[3];
        std::vector<unsigned short> a(9, 100), b(9, 200), c(9, 9999); // c is a gross outlier
        inputs[0] = a.data(); inputs[1] = b.data(); inputs[2] = c.data();
        std::vector<unsigned short> stacked(9, 0);
        Check(backend.MedianStack(inputs, 3, stacked.data(), 1, 3, 2.0f) == ComputeResult::SUCCESS,
              "CpuComputeBackend::MedianStack succeeds");
        Check(stacked[0] == 100 || stacked[0] == 200, "CpuComputeBackend::MedianStack rejects the gross outlier");
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

    // CpuComputeBackend only ever calls ONE tier at runtime (whichever
    // DetectCpuSimdTier() picks), so its own checks above never actually
    // exercise Kernels::Avx2::*/Kernels::Avx512::* unless the machine
    // running the test happens to have that hardware. This machine has
    // full AVX-512 (F/DQ/BW/VL), so call every tier directly here and
    // cross-check outputs against the Scalar tier - the only way to
    // actually verify the AVX2/AVX-512 kernels' correctness rather than
    // just their compilability. Tiers the running CPU doesn't support are
    // skipped (not failed) so this still runs safely on lesser hardware.
    void RunCrossTierSimdKernelChecks()
    {
        SimdTier tier = DetectCpuSimdTier();
        std::cout << "Detected CPU SIMD tier: " << SimdTierName(tier) << std::endl;
        bool hasAvx2 = (tier == SimdTier::Avx2 || tier == SimdTier::Avx512);
        bool hasAvx512 = (tier == SimdTier::Avx512);

        // MedianStack: sigma-clip + median is purely integer/exact-float
        // stats math - all three tiers must agree bit-for-bit.
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
            // A few gross outliers so sigma-clipping actually engages, not
            // just the plain-median path.
            for (int i = 0; i < 3; ++i) inputBuffers[i][100] = 60000;

            std::vector<unsigned short> outScalar(planeSize, 0);
            Kernels::Scalar::MedianStack(inputPtrs.data(), numInputs, outScalar.data(), width, height, 2.0f);

            if (hasAvx2)
            {
                std::vector<unsigned short> outAvx2(planeSize, 0);
                Kernels::Avx2::MedianStack(inputPtrs.data(), numInputs, outAvx2.data(), width, height, 2.0f);
                Check(outAvx2 == outScalar, "MedianStack: AVX2 tier bit-exact vs Scalar");
            }
            else std::cout << "SKIP: MedianStack AVX2 cross-check (CPU lacks AVX2)" << std::endl;

            if (hasAvx512)
            {
                std::vector<unsigned short> outAvx512(planeSize, 0);
                Kernels::Avx512::MedianStack(inputPtrs.data(), numInputs, outAvx512.data(), width, height, 2.0f);
                Check(outAvx512 == outScalar, "MedianStack: AVX512 tier bit-exact vs Scalar");
            }
            else std::cout << "SKIP: MedianStack AVX512 cross-check (CPU lacks AVX512)" << std::endl;
        }

        // WarpPerspective: floating-point bilinear blend - tiers are
        // allowed to differ by at most 1 ULP in the final 16-bit value
        // (documented FMA-contraction rounding noise in
        // WarpPerspectiveKernels.h), not bit-exact.
        {
            const int w = 40, h = 30;
            const size_t planeSize = static_cast<size_t>(w) * h * 3;
            std::vector<unsigned short> src(planeSize);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    for (int c = 0; c < 3; ++c)
                        src[(static_cast<size_t>(y) * w + x) * 3 + c] =
                            static_cast<unsigned short>((x * 37 + y * 53 + c * 17) % 4001);

            // Small rotation + translation, dest -> source (see the
            // "inverse homography" note in WarpPerspectiveKernels.h) -
            // non-integer sx/sy everywhere so the bilinear blend is
            // actually exercised, not just an exact-pixel copy.
            float angle = 0.06f;
            float ca = std::cos(angle), sa = std::sin(angle);
            float homography[9] = { ca, -sa, 1.7f, sa, ca, 0.9f, 0.0f, 0.0f, 1.0f };

            std::vector<unsigned short> outScalar(planeSize, 0);
            Kernels::Scalar::WarpPerspective(src.data(), w, h, outScalar.data(), w, h, homography, 0, 0);

            if (hasAvx2)
            {
                std::vector<unsigned short> outAvx2(planeSize, 0);
                Kernels::Avx2::WarpPerspective(src.data(), w, h, outAvx2.data(), w, h, homography, 0, 0);
                Check(MaxAbsDiff(outScalar, outAvx2) <= 1, "WarpPerspective: AVX2 tier within 1 ULP of Scalar");
            }
            else std::cout << "SKIP: WarpPerspective AVX2 cross-check (CPU lacks AVX2)" << std::endl;

            if (hasAvx512)
            {
                std::vector<unsigned short> outAvx512(planeSize, 0);
                Kernels::Avx512::WarpPerspective(src.data(), w, h, outAvx512.data(), w, h, homography, 0, 0);
                Check(MaxAbsDiff(outScalar, outAvx512) <= 1, "WarpPerspective: AVX512 tier within 1 ULP of Scalar");
            }
            else std::cout << "SKIP: WarpPerspective AVX512 cross-check (CPU lacks AVX512)" << std::endl;
        }

        // BayerDemosaic: BlackLevelSubtract/WhiteBalance are the only
        // vectorized stages (uniform elementwise arithmetic); demosaic and
        // color-matrix stay scalar in every tier - all three must agree
        // bit-for-bit.
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

            if (hasAvx2)
            {
                std::vector<unsigned short> outAvx2(static_cast<size_t>(w) * h * 3, 0);
                Kernels::Avx2::DemosaicBayer(cfa.data(), w, h, blackLevel, camMul, rgbCam, filters, outAvx2.data());
                Check(outAvx2 == outScalar, "BayerDemosaic: AVX2 tier bit-exact vs Scalar");
            }
            else std::cout << "SKIP: BayerDemosaic AVX2 cross-check (CPU lacks AVX2)" << std::endl;

            if (hasAvx512)
            {
                std::vector<unsigned short> outAvx512(static_cast<size_t>(w) * h * 3, 0);
                Kernels::Avx512::DemosaicBayer(cfa.data(), w, h, blackLevel, camMul, rgbCam, filters, outAvx512.data());
                Check(outAvx512 == outScalar, "BayerDemosaic: AVX512 tier bit-exact vs Scalar");
            }
            else std::cout << "SKIP: BayerDemosaic AVX512 cross-check (CPU lacks AVX512)" << std::endl;
        }
    }
}

int main()
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path tempDir = fs::temp_directory_path() / "image360_engine_smoke";
    fs::remove_all(tempDir, ec);
    fs::create_directories(tempDir, ec);
    Check(!ec, "create temp directory");

    fs::path dbPath = tempDir / "smoke.vfp";
    std::wstring wDbPath = Utf8ToWide(dbPath.string());
    std::wstring wProjectDir = Utf8ToWide(tempDir.string());

    {
        ProjectManager projectManager;
        Check(projectManager.CreateProject(wDbPath, 8192, 4096, 4096), "ProjectManager::CreateProject");

        StorageEngine storageEngine;
        Check(storageEngine.Open(wProjectDir, L"smoke", projectManager), "StorageEngine::Open");
        Check(storageEngine.IsOpen(), "StorageEngine::IsOpen after Open");

        std::vector<uint8_t> payload = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto blobId = storageEngine.WriteBlob(payload.data(), payload.size(), "test_blob");
        Check(blobId.has_value() && *blobId > 0, "StorageEngine::WriteBlob returns a valid blob id");

        if (blobId.has_value())
        {
            auto readBack = storageEngine.ReadBlob(*blobId);
            Check(readBack.has_value() && *readBack == payload, "StorageEngine::ReadBlob round-trips raw bytes exactly");
        }

        PixelBuffer buffer;
        buffer.width = 4;
        buffer.height = 2;
        buffer.data.assign(static_cast<size_t>(buffer.width) * buffer.height * 3, 0);
        for (size_t i = 0; i < buffer.data.size(); ++i)
            buffer.data[i] = static_cast<unsigned short>(i * 100);

        auto pixelBlobId = storageEngine.WritePixelBuffer(buffer, "raw_rgb48");
        Check(pixelBlobId.has_value(), "StorageEngine::WritePixelBuffer");

        if (pixelBlobId.has_value())
        {
            auto readBuffer = storageEngine.ReadPixelBuffer(*pixelBlobId);
            Check(readBuffer.has_value()
                      && readBuffer->width == buffer.width
                      && readBuffer->height == buffer.height
                      && readBuffer->data == buffer.data,
                  "StorageEngine::ReadPixelBuffer round-trips PixelBuffer exactly");
        }

        storageEngine.Close();
        projectManager.CloseProject();
    }

    {
        // Re-open with fresh instances to prove persistence across process
        // lifetimes, not just in-memory state.
        ProjectManager reopened;
        Check(reopened.LoadProject(wDbPath), "ProjectManager::LoadProject re-opens a persisted project");
        Check(reopened.GetTotalWidth() == 8192 && reopened.GetTotalHeight() == 4096,
              "Reloaded project metadata matches what was created");
    }

    fs::remove_all(tempDir, ec);

    RunRansacAndBundleAdjustmentChecks();
    RunCpuComputeBackendChecks();
    RunCrossTierSimdKernelChecks();

    if (g_failures == 0)
    {
        std::cout << "All engine smoke checks passed." << std::endl;
        return 0;
    }
    std::cerr << g_failures << " engine smoke check(s) failed." << std::endl;
    return 1;
}
