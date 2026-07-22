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
#include "HeaderFiles/PipelineDriver.h"
#include "HeaderFiles/ITaskExecutor.h"
#include "HeaderFiles/Tiff16Writer.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
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

    // Minimal ITaskExecutor for the burst-pipeline-foundation checks below
    // (docs/superpowers/plans/2026-07-21-burst-pipeline-foundation.md) -
    // WindowsApp.Tests/StubTaskExecutor.h isn't usable here (MSBuild-only
    // project), so this is a small self-contained equivalent: records which
    // stage each executed task belonged to (mutex-guarded - TaskScheduler
    // dispatches tasks within one stage concurrently) and always succeeds.
    class RecordingStubExecutor : public ITaskExecutor
    {
    public:
        explicit RecordingStubExecutor(PipelineStage stage) : m_stage(stage) {}

        bool Execute(Task& /*task*/, CancellationToken /*token*/) override
        {
            std::lock_guard<std::mutex> lock(s_sharedMutex);
            s_sharedOrder.push_back(m_stage);
            return true;
        }

        static std::vector<PipelineStage> TakeSharedOrder()
        {
            std::lock_guard<std::mutex> lock(s_sharedMutex);
            return s_sharedOrder;
        }
        static void ResetSharedOrder()
        {
            std::lock_guard<std::mutex> lock(s_sharedMutex);
            s_sharedOrder.clear();
        }

    private:
        PipelineStage m_stage;
        static inline std::mutex s_sharedMutex;
        static inline std::vector<PipelineStage> s_sharedOrder;
    };

    // docs/superpowers/plans/2026-07-21-burst-pipeline-foundation.md Task 4:
    // proves the Task/TaskScheduler/PipelineDriver machinery (already
    // production-proven for the panorama pipeline) generalizes to a second
    // pipeline family with zero new kernels - project-type round-trip plus
    // a real end-to-end burst stage sequence run via a stub executor.
    void RunBurstPipelineFoundationChecks()
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path tempDir = fs::temp_directory_path() / "image360_burst_foundation_smoke";
        fs::remove_all(tempDir, ec);
        fs::create_directories(tempDir, ec);

        // Step 2: panorama default stays default (no migration needed for
        // pre-existing .vfp files with no project_type/burst_mode key).
        {
            fs::path dbPath = tempDir / "panorama.vfp";
            ProjectManager panoramaProject;
            Check(panoramaProject.CreateProject(Utf8ToWide(dbPath.string()), 512, 256, 256),
                  "CreateProject (panorama) succeeds");
            Check(panoramaProject.GetProjectType() == ProjectType::PANORAMA,
                  "Panorama project reports ProjectType::PANORAMA");
            Check(panoramaProject.GetBurstMode() == BurstMode::NONE,
                  "Panorama project reports BurstMode::NONE");
        }

        // Step 1: burst project type/mode round-trips through a real
        // close+reopen, not just the in-memory value set at create time.
        {
            fs::path dbPath = tempDir / "burst.vfp";
            std::wstring wDbPath = Utf8ToWide(dbPath.string());
            {
                ProjectManager burstProject;
                Check(burstProject.CreateBurstProject(wDbPath, BurstMode::MFNR),
                      "CreateBurstProject(MFNR) succeeds");
                Check(burstProject.GetProjectType() == ProjectType::BURST,
                      "Burst project reports ProjectType::BURST immediately after create");
                Check(burstProject.GetBurstMode() == BurstMode::MFNR,
                      "Burst project reports BurstMode::MFNR immediately after create");
            }
            {
                ProjectManager reopened;
                Check(reopened.LoadProject(wDbPath), "LoadProject re-opens a persisted burst project");
                Check(reopened.GetProjectType() == ProjectType::BURST,
                      "Reloaded burst project still reports ProjectType::BURST");
                Check(reopened.GetBurstMode() == BurstMode::MFNR,
                      "Reloaded burst project still reports BurstMode::MFNR");
            }
        }

        // Step 3: the burst stage sequence (BURST_ALIGN -> BURST_MERGE ->
        // BURST_FINISH) actually runs end to end through PipelineDriver,
        // in order, with zero panorama-specific stages ever dispatched.
        {
            fs::path dbPath = tempDir / "burst_run.vfp";
            ProjectManager burstProject;
            Check(burstProject.CreateBurstProject(Utf8ToWide(dbPath.string()), BurstMode::HDR_PLUS),
                  "CreateBurstProject(HDR_PLUS) succeeds (run scenario)");

            Task alignTask;
            alignTask.stage = PipelineStage::BURST_ALIGN;
            alignTask.unitKind = "frame";
            alignTask.unitKey = "frame_0";

            Task mergeTask;
            mergeTask.stage = PipelineStage::BURST_MERGE;
            mergeTask.unitKind = "frame";
            mergeTask.unitKey = "frame_0";

            Task finishTask;
            finishTask.stage = PipelineStage::BURST_FINISH;
            finishTask.unitKind = "frame";
            finishTask.unitKey = "frame_0";

            Check(burstProject.CreateTasksIfAbsent({ alignTask, mergeTask, finishTask }),
                  "CreateTasksIfAbsent seeds one task per BURST_* stage");

            RecordingStubExecutor::ResetSharedOrder();
            PipelineDriver driver;
            driver.Initialize(nullptr, nullptr);
            driver.RegisterExecutor(PipelineStage::BURST_ALIGN, std::make_shared<RecordingStubExecutor>(PipelineStage::BURST_ALIGN));
            driver.RegisterExecutor(PipelineStage::BURST_MERGE, std::make_shared<RecordingStubExecutor>(PipelineStage::BURST_MERGE));
            driver.RegisterExecutor(PipelineStage::BURST_FINISH, std::make_shared<RecordingStubExecutor>(PipelineStage::BURST_FINISH));

            std::stop_source stopSource;
            bool ok = driver.Run(burstProject, stopSource.get_token());
            Check(ok, "PipelineDriver::Run completes the burst stage sequence");

            std::vector<PipelineStage> order = RecordingStubExecutor::TakeSharedOrder();
            std::vector<PipelineStage> expected = {
                PipelineStage::BURST_ALIGN, PipelineStage::BURST_MERGE, PipelineStage::BURST_FINISH
            };
            Check(order == expected, "Burst stages execute in order: BURST_ALIGN, BURST_MERGE, BURST_FINISH");

            for (PipelineStage stage : { PipelineStage::BURST_ALIGN, PipelineStage::BURST_MERGE, PipelineStage::BURST_FINISH })
            {
                std::vector<Task> tasks = burstProject.GetTasksForStage(stage);
                Check(tasks.size() == 1 && tasks[0].status == TaskStatus::COMPLETED,
                      "Burst stage task ends COMPLETED");
            }
        }

        fs::remove_all(tempDir, ec);
    }

    // Tiff16Writer round-trip check (docs/superpowers/plans/
    // 2026-07-22-cli-front-end.md Task 1) - no TIFF-reading library exists
    // in-tree, so this test IS the reader: it manually re-parses the
    // written bytes' header/IFD/tag values/pixel data against the TIFF6
    // spec's field layout, the same "the test is the only verification
    // this format is even correct" role WriteTiff16RGB's own header
    // comment describes.
    void RunTiff16WriterChecks()
    {
        std::cout << "\n=== Tiff16Writer round-trip ===" << std::endl;

        auto ReadU16 = [](const std::vector<unsigned char>& b, size_t off)
        {
            return static_cast<uint16_t>(b[off] | (static_cast<uint16_t>(b[off + 1]) << 8));
        };
        auto ReadU32 = [](const std::vector<unsigned char>& b, size_t off)
        {
            return static_cast<uint32_t>(b[off]) | (static_cast<uint32_t>(b[off + 1]) << 8)
                 | (static_cast<uint32_t>(b[off + 2]) << 16) | (static_cast<uint32_t>(b[off + 3]) << 24);
        };

        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path tempDir = fs::temp_directory_path() / "image360_tiff16_writer_smoke";
        fs::remove_all(tempDir, ec);
        fs::create_directories(tempDir, ec);
        Check(!ec, "create temp directory (Tiff16Writer)");

        constexpr int width = 3, height = 2;
        std::vector<unsigned short> pixels(static_cast<size_t>(width) * height * 3);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<unsigned short>(1000 + i * 777);

        fs::path tiffPath = tempDir / "test.tif";
        Check(WriteTiff16RGB(Utf8ToWide(tiffPath.string()), pixels.data(), width, height),
              "WriteTiff16RGB succeeds");

        std::ifstream file(tiffPath, std::ios::binary | std::ios::ate);
        Check(static_cast<bool>(file), "written TIFF file opens for reading");
        if (file)
        {
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<unsigned char> bytes(static_cast<size_t>(size));
            file.read(reinterpret_cast<char*>(bytes.data()), size);

            Check(bytes.size() >= 8, "file is at least large enough for a TIFF header");
            Check(bytes[0] == 'I' && bytes[1] == 'I', "byte order marker is little-endian \"II\"");
            Check(ReadU16(bytes, 2) == 42, "magic number is 42");

            uint32_t ifdOffset = ReadU32(bytes, 4);
            size_t pixelDataSize = static_cast<size_t>(width) * height * 3 * sizeof(unsigned short);
            Check(ifdOffset == 8 + pixelDataSize + 6 + 8 + 8,
                  "IFD offset lands after header + pixel data + BitsPerSample/XRes/YRes external blocks");

            // Pixel data starts immediately at offset 8 and is byte-identical.
            bool pixelsMatch = bytes.size() >= 8 + pixelDataSize
                && std::memcmp(bytes.data() + 8, pixels.data(), pixelDataSize) == 0;
            Check(pixelsMatch, "pixel data is written byte-identical starting at offset 8");

            if (ifdOffset + 2 <= bytes.size())
            {
                uint16_t entryCount = ReadU16(bytes, ifdOffset);
                Check(entryCount == 12, "IFD declares 12 entries");

                auto ReadTag = [&](int index) { return ReadU16(bytes, ifdOffset + 2 + static_cast<size_t>(index) * 12); };
                auto ReadValue = [&](int index) { return ReadU32(bytes, ifdOffset + 2 + static_cast<size_t>(index) * 12 + 8); };

                Check(ReadTag(0) == 256 && ReadValue(0) == static_cast<uint32_t>(width), "ImageWidth tag correct");
                Check(ReadTag(1) == 257 && ReadValue(1) == static_cast<uint32_t>(height), "ImageLength tag correct");
                Check(ReadTag(3) == 259 && ReadValue(3) == 1, "Compression tag is 1 (none)");
                Check(ReadTag(4) == 262 && ReadValue(4) == 2, "PhotometricInterpretation tag is 2 (RGB)");
                Check(ReadTag(5) == 273 && ReadValue(5) == 8, "StripOffsets tag points at offset 8");
                Check(ReadTag(6) == 277 && ReadValue(6) == 3, "SamplesPerPixel tag is 3");
                Check(ReadTag(7) == 278 && ReadValue(7) == static_cast<uint32_t>(height), "RowsPerStrip tag equals full height (single strip)");
                Check(ReadTag(8) == 279 && ReadValue(8) == static_cast<uint32_t>(pixelDataSize), "StripByteCounts tag matches pixel data size");

                // BitsPerSample (tag 258, external SHORT[3] block) - must be {16,16,16}.
                Check(ReadTag(2) == 258, "BitsPerSample tag id correct");
                uint32_t bpsOffset = ReadValue(2);
                bool bpsCorrect = bpsOffset + 6 <= bytes.size()
                    && ReadU16(bytes, bpsOffset) == 16 && ReadU16(bytes, bpsOffset + 2) == 16
                    && ReadU16(bytes, bpsOffset + 4) == 16;
                Check(bpsCorrect, "BitsPerSample external block is {16,16,16}");
            }
        }

        fs::remove_all(tempDir, ec);
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
    RunBurstPipelineFoundationChecks();
    RunTiff16WriterChecks();

    if (g_failures == 0)
    {
        std::cout << "All engine smoke checks passed." << std::endl;
        return 0;
    }
    std::cerr << g_failures << " engine smoke check(s) failed." << std::endl;
    return 1;
}
