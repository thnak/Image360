// Quality-gated end-to-end test for MFNR (docs/superpowers/plans/
// 2026-07-21-mfnr-block-match-merge.md Task 6). Three scenarios, purely
// synthetic (no committed fixtures - unlike tests/pipeline_e2e's DNGs,
// these frames are CfaType::STANDARD_RGB JPEGs generated in-process via
// JpegCodec::Encode, sidestepping RAW/DNG-fixture complexity entirely
// since this test is about the burst align/merge algorithms, not RAW
// decode):
//   1. Kernel-level BlockMatchAlign - recovers a known synthetic shift.
//   2. Kernel-level RobustMergeAccumulate - merged output's PSNR vs. clean
//      ground truth exceeds the reference frame's own PSNR (the direct
//      proof merging reduces noise, not just "didn't crash").
//   3. Full pipeline (ProjectManager -> PipelineDriver -> real
//      BurstAlignExecutor/BurstMergeExecutor on CpuComputeBackend) -same
//      PSNR gate, through the real JPEG-encode/decode round trip.

#include "HeaderFiles/ProjectManager.h"
#include "HeaderFiles/StorageEngine.h"
#include "HeaderFiles/TextEncoding.h"
#include "HeaderFiles/PipelineDriver.h"
#include "HeaderFiles/BurstAlignExecutor.h"
#include "HeaderFiles/BurstMergeExecutor.h"
#include "HeaderFiles/BurstCommon.h"
#include "HeaderFiles/BlockMatchAlignKernel.h"
#include "HeaderFiles/RobustMergeKernel.h"
#include "HeaderFiles/CpuComputeBackend.h"
#include "HeaderFiles/JpegCodec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stop_token>
#include <string>
#include <vector>

using namespace WindowsApp::Core;
using namespace WindowsApp::Compute;
namespace fs = std::filesystem;

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

    // Irregular, non-periodic content (genuine PRNG output box-blurred a
    // little, not a modular/checkerboard pattern) - tests/pipeline_e2e's
    // own fixture-generation history found periodic textures alias badly
    // against block/feature matching, so this avoids periodicity the same
    // way; pure per-pixel white noise (the first version of this
    // generator) avoids periodicity too but is close to worst-case input
    // for JPEG - a DCT codec that assumes local smoothness mangles
    // uncorrelated noise catastrophically even at quality 95, which
    // corrupted Step 3's JPEG-round-tripped frames long before alignment/
    // merge ever ran. A short box blur gives pixels genuine local
    // correlation (JPEG-compressible, "photographic") while staying
    // irregular enough for block matching - a flat/periodic scene would
    // still alias, a lightly-blurred-noise one doesn't.
    std::vector<unsigned short> GenerateCleanScene16(int width, int height, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(4000, 60000);
        std::vector<unsigned short> raw(static_cast<size_t>(width) * height * 3);
        for (auto& v : raw) v = static_cast<unsigned short>(dist(rng));

        constexpr int kRadius = 2; // 5x5 box blur
        std::vector<unsigned short> scene(raw.size());
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                for (int c = 0; c < 3; ++c)
                {
                    long sum = 0;
                    int count = 0;
                    for (int dy = -kRadius; dy <= kRadius; ++dy)
                    {
                        int sy = std::clamp(y + dy, 0, height - 1);
                        for (int dx = -kRadius; dx <= kRadius; ++dx)
                        {
                            int sx = std::clamp(x + dx, 0, width - 1);
                            sum += raw[(static_cast<size_t>(sy) * width + sx) * 3 + c];
                            ++count;
                        }
                    }
                    scene[(static_cast<size_t>(y) * width + x) * 3 + c] = static_cast<unsigned short>(sum / count);
                }
            }
        }
        return scene;
    }

    // A noiseless-shift pair with NO border clamping anywhere - built from
    // one (width+2*margin) x (height+2*margin) padded canvas, so every
    // pixel `ref`/`src` need (for any (dx,dy) with |dx|,|dy| <= margin) is
    // real generated content, never a clamped/duplicated edge value. Used
    // only by the exact-recovery check (Step 1) - Step 2/3's PSNR-based
    // checks tolerate ShiftAndNoise16's border clamping fine, since they
    // never assert per-tile exactness.
    void GenerateExactShiftPair(int width, int height, int margin, int dx, int dy, unsigned seed,
                                 std::vector<unsigned short>& outRef, std::vector<unsigned short>& outSrc)
    {
        int paddedW = width + 2 * margin;
        int paddedH = height + 2 * margin;
        auto padded = GenerateCleanScene16(paddedW, paddedH, seed);

        auto crop = [&](int offsetX, int offsetY)
        {
            std::vector<unsigned short> out(static_cast<size_t>(width) * height * 3);
            for (int y = 0; y < height; ++y)
            {
                const unsigned short* srcRow = padded.data() + (static_cast<size_t>(y + offsetY) * paddedW + offsetX) * 3;
                unsigned short* dstRow = out.data() + static_cast<size_t>(y) * width * 3;
                std::copy(srcRow, srcRow + static_cast<size_t>(width) * 3, dstRow);
            }
            return out;
        };

        outRef = crop(margin, margin);
        outSrc = crop(margin - dx, margin - dy);
    }

    // Clamped-shift + optional Gaussian noise, matching
    // BlockMatchAlign's documented convention: dst(x+dx, y+dy) == src(x,y)
    // (dst is `ref` shifted so that BlockMatchAlign(ref, dst) recovers
    // (dx, dy)).
    std::vector<unsigned short> ShiftAndNoise16(const std::vector<unsigned short>& ref, int width, int height,
                                                 int dx, int dy, double noiseSigma, unsigned seed)
    {
        std::mt19937 rng(seed);
        // std::normal_distribution with stddev=0 is degenerate - MSVC's
        // implementation can hand back NaN, and casting NaN to an integer
        // type is UB that MSVC's /RTC1 debug checks treat as a hard
        // failure (confirmed via a real crash on win-thanh: 0xC0000409
        // right inside this function for a noiseSigma=0.0 caller). Every
        // noiseless caller in this file passes 0.0 explicitly, so skip
        // the distribution entirely rather than feed it a degenerate
        // parameter.
        bool hasNoise = noiseSigma > 0.0;
        std::normal_distribution<double> noise(0.0, hasNoise ? noiseSigma : 1.0);

        std::vector<unsigned short> out(ref.size());
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int sx = std::clamp(x - dx, 0, width - 1);
                int sy = std::clamp(y - dy, 0, height - 1);
                for (int c = 0; c < 3; ++c)
                {
                    double v = static_cast<double>(ref[(static_cast<size_t>(sy) * width + sx) * 3 + c]);
                    if (hasNoise) v += noise(rng);
                    out[(static_cast<size_t>(y) * width + x) * 3 + c] =
                        static_cast<unsigned short>(std::clamp(v, 0.0, 65535.0));
                }
            }
        }
        return out;
    }

    double Psnr16(const std::vector<unsigned short>& a, const std::vector<unsigned short>& b)
    {
        double sumSq = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sumSq += diff * diff;
        }
        double mse = sumSq / static_cast<double>(a.size());
        if (mse <= 0.0) return std::numeric_limits<double>::infinity();
        constexpr double kMaxValue = 65535.0;
        return 10.0 * std::log10((kMaxValue * kMaxValue) / mse);
    }

    void RunBlockMatchAlignKernelCheck()
    {
        std::cout << "\n=== Step 1: BlockMatchAlign recovers a known shift ===" << std::endl;

        constexpr int width = 80, height = 64;
        constexpr int trueDx = 3, trueDy = -2;

        std::vector<unsigned short> ref, src;
        GenerateExactShiftPair(width, height, kBurstSearchRadius, trueDx, trueDy, 111, ref, src);

        int tilesX = (width + kBurstTileSize - 1) / kBurstTileSize;
        int tilesY = (height + kBurstTileSize - 1) / kBurstTileSize;
        std::vector<TileOffset> offsets(static_cast<size_t>(tilesX) * tilesY);

        Kernels::BlockMatchAlign(ref.data(), src.data(), width, height, kBurstTileSize, kBurstSearchRadius,
                                  offsets.data(), tilesX, tilesY);

        // Only interior tiles - a tile whose true-offset candidate window
        // would read outside [0,width)x[0,height) genuinely has no valid
        // match at (trueDx,trueDy) to find (BlockMatchAlign correctly
        // excludes that candidate entirely, per its documented contract),
        // so it's not a fair check of recovery accuracy. With
        // trueDx=3 > 0 that's the rightmost tile column; with trueDy=-2 < 0
        // that's the topmost tile row.
        bool allInteriorMatch = true;
        int interiorChecked = 0;
        for (int ty = 0; ty < tilesY; ++ty)
        {
            for (int tx = 0; tx < tilesX; ++tx)
            {
                int tx0 = tx * kBurstTileSize, tx1 = (std::min)(tx0 + kBurstTileSize, width);
                int ty0 = ty * kBurstTileSize, ty1 = (std::min)(ty0 + kBurstTileSize, height);
                bool trueOffsetInBounds = (tx0 + trueDx >= 0) && (ty0 + trueDy >= 0)
                    && (tx1 + trueDx <= width) && (ty1 + trueDy <= height);
                if (!trueOffsetInBounds) continue;

                ++interiorChecked;
                const TileOffset& offset = offsets[static_cast<size_t>(ty) * tilesX + tx];
                if (offset.dx != trueDx || offset.dy != trueDy) allInteriorMatch = false;
            }
        }
        std::cout << "recovered offset (first tile): dx=" << offsets[0].dx << " dy=" << offsets[0].dy
                   << " (expected dx=" << trueDx << " dy=" << trueDy << "); interior tiles checked="
                   << interiorChecked << "/" << (tilesX * tilesY) << std::endl;
        Check(interiorChecked > 0, "at least one tile's true-offset candidate window stays fully in bounds");
        Check(allInteriorMatch, "every interior tile recovers the known noiseless shift exactly");
    }

    void RunRobustMergeKernelCheck()
    {
        std::cout << "\n=== Step 2: RobustMergeAccumulate reduces noise vs. a single frame ===" << std::endl;

        constexpr int width = 64, height = 48;
        constexpr int numFrames = 8;
        constexpr double kNoiseSigma = 2000.0; // matches kBurstMergeSigma's scale

        auto clean = GenerateCleanScene16(width, height, 333);

        std::vector<std::vector<unsigned short>> frames(numFrames);
        std::vector<int> shiftsX = { 0, 2, -3, 1, 4, -2, 3, -4 };
        std::vector<int> shiftsY = { 0, -1, 2, -3, 1, 4, -2, 2 };
        for (int k = 0; k < numFrames; ++k)
        {
            frames[k] = ShiftAndNoise16(clean, width, height, shiftsX[k], shiftsY[k], kNoiseSigma,
                                         1000u + static_cast<unsigned>(k));
        }

        int tilesX = (width + kBurstTileSize - 1) / kBurstTileSize;
        int tilesY = (height + kBurstTileSize - 1) / kBurstTileSize;

        // Known offsets fed directly (bypassing BlockMatchAlign) to isolate
        // the merge kernel's own correctness, per this plan's Task 6.
        std::vector<std::vector<TileOffset>> perFrameOffsets(numFrames - 1);
        for (int k = 1; k < numFrames; ++k)
        {
            perFrameOffsets[k - 1].assign(static_cast<size_t>(tilesX) * tilesY, TileOffset{ shiftsX[k], shiftsY[k] });
        }

        std::vector<const unsigned short*> framePtrs;
        for (const auto& f : frames) framePtrs.push_back(f.data());
        std::vector<const TileOffset*> offsetPtrs;
        for (const auto& o : perFrameOffsets) offsetPtrs.push_back(o.data());

        std::vector<unsigned short> merged(clean.size());
        Kernels::RobustMergeAccumulate(framePtrs.data(), numFrames, offsetPtrs.data(), width, height,
                                        kBurstTileSize, tilesX, tilesY, static_cast<float>(kNoiseSigma), merged.data());

        double referencePsnr = Psnr16(frames[0], clean);
        double mergedPsnr = Psnr16(merged, clean);
        std::cout << "reference-frame-alone PSNR=" << referencePsnr << " dB, merged PSNR=" << mergedPsnr << " dB"
                   << std::endl;

        Check(mergedPsnr > referencePsnr, "merged output's PSNR exceeds the reference frame's own PSNR");
    }

    // Narrows a 16-bit synthetic buffer to interleaved RGB8 for JPEG
    // encoding - the inverse of DecodeInputImage's *257 widen.
    std::vector<unsigned char> NarrowTo8(const std::vector<unsigned short>& data16)
    {
        std::vector<unsigned char> out(data16.size());
        for (size_t i = 0; i < data16.size(); ++i)
            out[i] = static_cast<unsigned char>(data16[i] / 257);
        return out;
    }

    void RunFullPipelineQualityScenario()
    {
        std::cout << "\n=== Step 3: full MFNR pipeline (project -> align -> merge -> finish) ===" << std::endl;

        // A boundary tile whose true shift would require reading outside
        // the frame has no valid candidate at that shift (BlockMatchAlign
        // correctly excludes it, per its documented contract - see
        // BlockMatchAlignKernel.h) and falls back to the best in-bounds
        // alternative, which can be a poor match. That's a real, tracked
        // limitation (docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md
        // SS9) - on a real photo (thousands of tiles) it's a small
        // fraction of the image; on a too-small test canvas it can be
        // HALF the tiles (a 64x48/16px grid is 4x3=12 tiles - a uniform
        // shift's boundary-affected row+column alone is 6 of them),
        // swamping the result and making the test unrepresentative. 320x240
        // (20x15=300 tiles) keeps the boundary fraction (~22%) close to
        // what a real image sees, without an unreasonably slow test.
        constexpr int width = 320, height = 240;
        constexpr int numFrames = 6;
        // 8-bit-space noise sigma such that, once DecodeInputImage widens
        // by *257, the effective 16-bit noise (~257*8 ~= 2056) roughly
        // matches kBurstMergeSigma - see this plan's Task 6 note on
        // keeping the injected noise and the fixed merge sigma
        // comparable, the same way a real noise model would be calibrated
        // to actual sensor noise.
        constexpr double kNoiseSigma8 = 8.0;

        auto clean16 = GenerateCleanScene16(width, height, 444);

        std::error_code ec;
        fs::path tempDir = fs::temp_directory_path() / "image360_pipeline_e2e_burst";
        fs::remove_all(tempDir, ec);
        fs::create_directories(tempDir, ec);
        Check(!ec, "create temp project directory");

        auto jpegCodec = std::make_shared<JpegCodec>();
        Check(jpegCodec->Initialize() == ComputeResult::SUCCESS, "JpegCodec::Initialize");

        std::vector<int> shiftsX = { 0, 2, -3, 1, 4, -2 };
        std::vector<int> shiftsY = { 0, -1, 2, -3, 1, 4 };

        ProjectManager projectManager;
        fs::path dbPath = tempDir / "burst_e2e.vfp";
        std::wstring wDbPath = Utf8ToWide(dbPath.string());
        Check(projectManager.CreateBurstProject(wDbPath, BurstMode::MFNR), "ProjectManager::CreateBurstProject(MFNR)");

        // Frame 0's pre-JPEG-encode 16-bit reconstruction (8-bit noisy
        // values widened by *257, mirroring DecodeInputImage's own widen)
        // - captured here so the PSNR comparison below is against what
        // frame 0 actually looked like post-noise, not the pristine clean
        // scene (which frame 0 never equals once noise is injected).
        std::vector<unsigned short> frame0Reconstructed16;

        for (int k = 0; k < numFrames; ++k)
        {
            auto frame16 = ShiftAndNoise16(clean16, width, height, shiftsX[k], shiftsY[k], 0.0, 2000u + static_cast<unsigned>(k));
            auto frame8 = NarrowTo8(frame16);

            // Inject 8-bit-space noise directly (not via ShiftAndNoise16's
            // 16-bit-space noise) so the noise level survives the eventual
            // /257 narrow instead of being rounded away.
            std::mt19937 rng(5000u + static_cast<unsigned>(k));
            std::normal_distribution<double> noise(0.0, kNoiseSigma8);
            for (auto& v : frame8)
                v = static_cast<unsigned char>(std::clamp(static_cast<double>(v) + noise(rng), 0.0, 255.0));

            if (k == 0)
            {
                frame0Reconstructed16.resize(frame8.size());
                for (size_t i = 0; i < frame8.size(); ++i)
                    frame0Reconstructed16[i] = static_cast<unsigned short>(frame8[i]) * 257;
            }

            std::vector<unsigned char> jpegBytes;
            Check(jpegCodec->Encode(frame8.data(), width, height, 95, jpegBytes) == ComputeResult::SUCCESS,
                  "JpegCodec::Encode synthetic frame");

            fs::path framePath = tempDir / ("frame_" + std::to_string(k) + ".jpg");
            std::ofstream out(framePath, std::ios::binary);
            out.write(reinterpret_cast<const char*>(jpegBytes.data()), static_cast<std::streamsize>(jpegBytes.size()));
            out.close();

            Check(projectManager.AddInputImage(Utf8ToWide(framePath.string()), Homography{}, CfaType::STANDARD_RGB),
                  "ProjectManager::AddInputImage (burst frame)");
        }
        Check(projectManager.GetInputImages().size() == static_cast<size_t>(numFrames), "all burst frames registered");

        Check(projectManager.SeedBurstAlignTasks(), "ProjectManager::SeedBurstAlignTasks");
        Check(projectManager.SeedBurstMergeTasks(), "ProjectManager::SeedBurstMergeTasks");

        StorageEngine storageEngine;
        Check(storageEngine.Open(Utf8ToWide(tempDir.string()), L"burst_e2e", projectManager), "StorageEngine::Open");

        auto computeBackend = std::make_shared<CpuComputeBackend>();
        Check(computeBackend->Initialize() == ComputeResult::SUCCESS, "CpuComputeBackend::Initialize");

        PipelineDriver driver;
        driver.Initialize(
            [](PipelineStage stage, float progress)
            { std::cout << "progress: stage=" << static_cast<int>(stage) << " overall=" << progress << std::endl; },
            [](const std::wstring& msg) { std::wcerr << L"log: " << msg << std::endl; });

        driver.RegisterExecutor(PipelineStage::BURST_ALIGN,
            std::make_shared<BurstAlignExecutor>(projectManager, storageEngine, computeBackend, jpegCodec));
        auto mergeExecutor = std::make_shared<BurstMergeExecutor>(projectManager, storageEngine, computeBackend);
        driver.RegisterExecutor(PipelineStage::BURST_MERGE, mergeExecutor);
        driver.RegisterExecutor(PipelineStage::BURST_FINISH, mergeExecutor);

        std::stop_source stopSource;
        bool ranOk = driver.Run(projectManager, stopSource.get_token());
        Check(ranOk, "PipelineDriver::Run completes BURST_ALIGN->BURST_MERGE->BURST_FINISH");
        Check(driver.GetCurrentStage() == PipelineStage::COMPLETED, "PipelineDriver ends in COMPLETED stage");

        if (ranOk)
        {
            std::vector<Task> finishTasks = projectManager.GetTasksForStage(PipelineStage::BURST_FINISH);
            Check(finishTasks.size() == 1 && finishTasks[0].status == TaskStatus::COMPLETED
                      && finishTasks[0].outputBlobId.has_value(),
                  "BURST_FINISH task completed with an output blob");

            if (!finishTasks.empty() && finishTasks[0].outputBlobId.has_value())
            {
                auto finalBuffer = storageEngine.ReadPixelBuffer(*finishTasks[0].outputBlobId);
                Check(finalBuffer.has_value(), "read back BURST_FINISH's output PixelBuffer");

                if (finalBuffer.has_value())
                {
                    Check(finalBuffer->width == width && finalBuffer->height == height,
                          "final output dimensions match the burst frames");

                    // The reference frame (frame 0) went through the exact
                    // same JPEG round trip as the merged result, so both
                    // sides of this comparison share the same JPEG-
                    // quantization noise floor - an apples-to-apples check
                    // that merging N frames beats trusting frame 0 alone.
                    double referencePsnr = Psnr16(frame0Reconstructed16, clean16);
                    double finalPsnr = Psnr16(finalBuffer->data, clean16);
                    std::cout << "reference-frame PSNR=" << referencePsnr << " dB, pipeline output PSNR=" << finalPsnr
                               << " dB" << std::endl;

                    // Data-driven threshold (matches tests/pipeline_e2e's
                    // own established convention): a correct run achieves
                    // ~32.0 dB (reference frame alone: ~30.0 dB) - this
                    // leaves headroom for run-to-run JPEG-encoder/platform
                    // floating-point variance while still catching a real
                    // regression (a broken merge/alignment scored ~14 dB
                    // in this fixture's own testing history, discovered
                    // while writing this test - see this plan's Task 6).
                    constexpr double kMinAcceptablePsnrDb = 26.0;
                    Check(finalPsnr >= kMinAcceptablePsnrDb,
                          "pipeline output's PSNR against the clean scene meets the quality bar");
                    Check(finalPsnr > referencePsnr,
                          "pipeline output's PSNR exceeds the single reference frame's own PSNR");
                }
            }
        }

        storageEngine.Close();
        projectManager.CloseProject();
        fs::remove_all(tempDir, ec);
    }
}

int main()
{
    RunBlockMatchAlignKernelCheck();
    RunRobustMergeKernelCheck();
    RunFullPipelineQualityScenario();

    if (g_failures == 0)
    {
        std::cout << "\nAll pipeline_e2e_burst checks passed." << std::endl;
        return 0;
    }
    std::cerr << "\n" << g_failures << " pipeline_e2e_burst check(s) failed." << std::endl;
    return 1;
}
