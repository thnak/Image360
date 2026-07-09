// Full pipeline integration test: loads 3 synthetic RAW (DNG) tiles cropped
// with known overlap from one master scene (see
// scripts/panorama_gen/gen_scene.py - tile_width=320, stride=208, so tiles
// sit at world-space x_offset 0/208/416), drives PipelineDriver through all
// 4 stages (Ingest -> Align -> Optimize -> Render) on the CPU compute
// backend, and checks that Optimize's bundle adjustment actually recovers
// those known offsets - not just that the pipeline runs without crashing.
//
// Three scenarios share the same fixtures but use different chunkSize
// values or verify different things:
//   1. A single chunk covering the whole canvas - verifies the recovered
//      geometry itself (homographies match known tile offsets).
//   2. A small chunkSize forcing a 3-chunk grid over the same 3 images -
//      verifies Render genuinely computes "piece by piece": each chunk
//      gets a DIFFERENT contributor subset (not all 3 images trivially,
//      like scenario 1's single chunk does) and each chunk's rendered
//      PixelBuffer comes back at that chunk's own (non-uniform) width,
//      not the whole canvas.
//   3. Re-runs scenario 1's single-chunk project, exports the assembled
//      panorama via PanoramaExporter, and compares it pixel-for-pixel
//      against tests/pipeline_e2e/fixtures/scene_reference.jpg - the
//      exact same master scene the 3 tiles were cropped from (regenerate
//      both together via scripts/panorama_gen/generate_all.py) - via MSE/
//      PSNR, so "did the reconstructed panorama actually look like the
//      source" has a number attached, not just "did every stage return
//      true".
//
// The fixture DNGs are hand-rolled (see scripts/panorama_gen/make_dng.py)
// but exercise the exact same ImageLoader/LibRaw code path a real camera
// RAW file would: Bayer CFA unpack for Ingest, embedded JPEG preview for
// Align/Optimize's gain step.

#include "HeaderFiles/ProjectManager.h"
#include "HeaderFiles/StorageEngine.h"
#include "HeaderFiles/TextEncoding.h"
#include "HeaderFiles/PipelineDriver.h"
#include "HeaderFiles/RawIngestExecutor.h"
#include "HeaderFiles/AlignExecutor.h"
#include "HeaderFiles/OptimizeExecutor.h"
#include "HeaderFiles/RenderExecutor.h"
#include "HeaderFiles/PanoramaExporter.h"
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
#include <stop_token>
#include <thread>
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

    struct TileGroundTruth
    {
        const wchar_t* fileName;
        float worldXOffset; // known crop offset relative to tile_0's frame
    };

    // Must match scripts/panorama_gen/gen_scene.py's TILE_W/STRIDE and the
    // committed fixtures/tile_*.dng - regenerate both together via
    // scripts/panorama_gen/generate_all.py if these ever change.
    constexpr int kTileWidth = 320;
    constexpr int kTileHeight = 240;
    constexpr TileGroundTruth kTiles[] = {
        { L"tile_0.dng", 0.0f },
        { L"tile_1.dng", 208.0f },
        { L"tile_2.dng", 416.0f },
    };
    constexpr float kOffsetTolerancePx = 6.0f; // demosaic/JPEG/feature-match noise budget

    namespace fs = std::filesystem;

    // Builds a fresh project from the 3 fixture tiles and drives the full
    // pipeline to completion. Leaves projectManager/storageEngine open so
    // the caller can inspect the result (homographies, chunks, tasks).
    bool SetupAndRunPipeline(const fs::path& tempDir, const fs::path& fixturesDir,
                              int totalWidth, int totalHeight, int chunkSize,
                              ProjectManager& projectManager, StorageEngine& storageEngine)
    {
        fs::path dbPath = tempDir / "e2e.vfp";
        std::wstring wDbPath = Utf8ToWide(dbPath.string());
        std::wstring wProjectDir = Utf8ToWide(tempDir.string());

        Check(projectManager.CreateProject(wDbPath, totalWidth, totalHeight, chunkSize),
              "ProjectManager::CreateProject");

        for (const auto& tile : kTiles)
        {
            fs::path tilePath = fixturesDir / tile.fileName;
            Check(fs::exists(tilePath), "fixture DNG exists");
            std::wstring wTilePath = Utf8ToWide(tilePath.string());
            Check(projectManager.AddInputImage(wTilePath, Homography{}, CfaType::BAYER),
                  "ProjectManager::AddInputImage");
        }
        Check(projectManager.GetInputImages().size() == 3, "all 3 input images registered");

        Check(storageEngine.Open(wProjectDir, L"e2e", projectManager), "StorageEngine::Open");

        // PipelineDriver::Run only seeds STAGE3_RENDER's tasks itself (they
        // depend on Optimize's final homographies) - Ingest/Align/Optimize
        // must be seeded upfront by the caller (see PipelineDriver.cpp's
        // own comment on this split).
        Check(projectManager.SeedIngestTasks(), "ProjectManager::SeedIngestTasks");
        Check(projectManager.SeedAlignTasks(), "ProjectManager::SeedAlignTasks");
        Check(projectManager.SeedOptimizeTasks(), "ProjectManager::SeedOptimizeTasks");

        auto computeBackend = std::make_shared<CpuComputeBackend>();
        Check(computeBackend->Initialize() == ComputeResult::SUCCESS, "CpuComputeBackend::Initialize");

        auto jpegCodec = std::make_shared<JpegCodec>();
        Check(jpegCodec->Initialize() == ComputeResult::SUCCESS, "JpegCodec::Initialize");

        PipelineDriver driver;
        size_t maxInFlight = std::max<size_t>(1, std::thread::hardware_concurrency() - 1);
        driver.Initialize(
            [](PipelineStage stage, float progress)
            {
                std::cout << "progress: stage=" << static_cast<int>(stage) << " overall=" << progress << std::endl;
            },
            [](const std::wstring& msg)
            {
                std::wcerr << L"log: " << msg << std::endl;
            },
            maxInFlight);

        driver.RegisterExecutor(PipelineStage::STAGE0_INGEST,
            std::make_shared<RawIngestExecutor>(projectManager, storageEngine, computeBackend));
        driver.RegisterExecutor(PipelineStage::STAGE1_ALIGN,
            std::make_shared<AlignExecutor>(projectManager, storageEngine, computeBackend, jpegCodec));
        driver.RegisterExecutor(PipelineStage::STAGE2_OPTIMIZE,
            std::make_shared<OptimizeExecutor>(projectManager, storageEngine, computeBackend, jpegCodec));
        driver.RegisterExecutor(PipelineStage::STAGE3_RENDER,
            std::make_shared<RenderExecutor>(projectManager, storageEngine, computeBackend));

        std::stop_source stopSource;
        bool ranOk = driver.Run(projectManager, stopSource.get_token());
        Check(ranOk, "PipelineDriver::Run completes Ingest->Align->Optimize->Render");
        Check(driver.GetCurrentStage() == PipelineStage::COMPLETED, "PipelineDriver ends in COMPLETED stage");
        return ranOk;
    }

    void RunSingleChunkScenario(const fs::path& fixturesDir)
    {
        std::cout << "\n=== Scenario 1: single chunk covering the whole canvas ===" << std::endl;

        std::error_code ec;
        fs::path tempDir = fs::temp_directory_path() / "image360_pipeline_e2e_singlechunk";
        fs::remove_all(tempDir, ec);
        fs::create_directories(tempDir, ec);
        Check(!ec, "create temp project directory (single-chunk scenario)");

        ProjectManager projectManager;
        StorageEngine storageEngine;
        // World canvas must cover every tile's full extent once placed at
        // its true offset (rightmost tile: 416 + 320 = 736 wide, 240
        // tall) - chunkSize=1024 collapses to a single "C_0_0" chunk
        // spanning the whole canvas.
        bool ranOk = SetupAndRunPipeline(tempDir, fixturesDir, 800, 280, 1024, projectManager, storageEngine);

        if (ranOk)
        {
            // The real payoff: did Optimize's bundle adjustment actually
            // recover the known geometric relationship between the tiles,
            // not just avoid crashing? BundleAdjustment's homography
            // convention (confirmed via
            // WindowsApp.Tests/BundleAdjustmentTests.cpp) maps an image's
            // own local pixel coordinates into the shared world/reference
            // frame - the reference image (smallest id, tile_0) is fixed
            // at identity, so every other tile's h[2] (x-translation)
            // should land close to its known world offset.
            const auto& finalImages = projectManager.GetInputImages();
            for (size_t i = 0; i < finalImages.size() && i < std::size(kTiles); ++i)
            {
                const auto& img = finalImages[i];
                float dx = img.homography.h[2];
                float dy = img.homography.h[5];
                std::cout << "image id=" << img.id << " recovered h02(dx)=" << dx << " h12(dy)=" << dy
                           << " expected dx=" << kTiles[i].worldXOffset << std::endl;

                Check(std::fabs(dx - kTiles[i].worldXOffset) < kOffsetTolerancePx,
                      "recovered x-translation matches known tile offset");
                Check(std::fabs(dy) < kOffsetTolerancePx,
                      "recovered y-translation stays near zero (tiles have no vertical shift)");
            }

            const auto& chunks = projectManager.GetChunks();
            Check(chunks.size() == 1, "exactly one render chunk for this canvas/chunkSize");
            if (!chunks.empty())
            {
                auto contributors = projectManager.GetChunkContributors(chunks.front().id);
                Check(contributors.size() == 3, "all 3 tiles contribute to the single chunk");
            }

            std::vector<Task> renderTasks = projectManager.GetTasksForStage(PipelineStage::STAGE3_RENDER);
            Check(!renderTasks.empty(), "at least one Render task was seeded");
            Check(std::all_of(renderTasks.begin(), renderTasks.end(),
                      [](const Task& t) { return t.status == TaskStatus::COMPLETED && t.outputBlobId.has_value(); }),
                  "every Render task completed with a written output blob");
        }

        storageEngine.Close();
        projectManager.CloseProject();
        fs::remove_all(tempDir, ec);
    }

    void RunMultiChunkScenario(const fs::path& fixturesDir)
    {
        std::cout << "\n=== Scenario 2: multi-chunk grid (\"compute piece by piece\") ===" << std::endl;

        std::error_code ec;
        fs::path tempDir = fs::temp_directory_path() / "image360_pipeline_e2e_multichunk";
        fs::remove_all(tempDir, ec);
        fs::create_directories(tempDir, ec);
        Check(!ec, "create temp project directory (multi-chunk scenario)");

        ProjectManager projectManager;
        StorageEngine storageEngine;

        // Same 800x280 canvas as scenario 1, but chunkSize=300 forces a
        // 3x1 grid (chunk widths 300/300/200, ProjectManager::CreateProject
        // clamps the last chunk's width to what's left of totalWidth) -
        // genuinely exercises per-chunk overlap culling and per-chunk
        // Render dispatch instead of trivially covering everything in one
        // chunk. Expected per-chunk contributors, derived from the known
        // tile placements (id1 x:[0,320), id2 x:[208,528), id3
        // x:[416,736)) intersected against each chunk's x-range:
        //   C_0_0 x:[0,300)   -> {id1,id2}      (id3 starts at 416, no overlap)
        //   C_1_0 x:[300,600) -> {id1,id2,id3}  (id1's tail [300,320) just barely overlaps)
        //   C_2_0 x:[600,800) -> {id3}          (id1/id2 both end before 600)
        constexpr int kChunkSize = 300;
        bool ranOk = SetupAndRunPipeline(tempDir, fixturesDir, 800, 280, kChunkSize, projectManager, storageEngine);

        if (ranOk)
        {
            const auto& chunks = projectManager.GetChunks();
            Check(chunks.size() == 3, "800x280 canvas with chunkSize=300 produces exactly 3 chunks");

            struct ExpectedChunk
            {
                const char* id;
                int expectedWidth;
                size_t expectedContributorCount;
            };
            constexpr ExpectedChunk kExpected[] = {
                { "C_0_0", 300, 2 },
                { "C_1_0", 300, 3 },
                { "C_2_0", 200, 1 }, // 800 - 600, clamped by CreateProject
            };

            for (const auto& expected : kExpected)
            {
                auto it = std::find_if(chunks.begin(), chunks.end(),
                    [&](const ChunkModel& c) { return c.id == expected.id; });
                bool found = it != chunks.end();
                Check(found, "expected chunk id exists in the grid");
                if (!found) continue;

                Check(it->width == expected.expectedWidth, "chunk width matches expected grid geometry");
                Check(it->height == 280, "chunk height matches the (single-row) canvas height");

                auto contributors = projectManager.GetChunkContributors(it->id);
                std::cout << "chunk " << it->id << " (x=" << it->x_offset << " w=" << it->width << "): "
                           << contributors.size() << " contributor(s), expected " << expected.expectedContributorCount
                           << std::endl;
                Check(contributors.size() == expected.expectedContributorCount,
                      "chunk has the expected number of contributing images");
            }

            // The actual "piece by piece" proof: each Render task's
            // committed output must come back at ITS OWN chunk's
            // dimensions, not the whole canvas - if Render silently
            // processed everything as one blob, every task's PixelBuffer
            // would share one (wrong) size instead of each matching its
            // own chunk.
            std::vector<Task> renderTasks = projectManager.GetTasksForStage(PipelineStage::STAGE3_RENDER);
            Check(renderTasks.size() == 3, "exactly 3 Render tasks seeded (one per non-empty chunk)");

            bool allDimsMatch = true;
            for (const auto& task : renderTasks)
            {
                auto chunkIt = std::find_if(chunks.begin(), chunks.end(),
                    [&](const ChunkModel& c) { return c.id == task.unitKey; });
                if (chunkIt == chunks.end() || task.status != TaskStatus::COMPLETED || !task.outputBlobId.has_value())
                {
                    allDimsMatch = false;
                    continue;
                }
                auto rendered = storageEngine.ReadPixelBuffer(task.outputBlobId.value());
                if (!rendered.has_value() ||
                    rendered->width != chunkIt->width || rendered->height != chunkIt->height)
                {
                    allDimsMatch = false;
                    std::cerr << "chunk " << task.unitKey << " rendered buffer mismatch" << std::endl;
                }
            }
            Check(allDimsMatch, "every rendered chunk's output PixelBuffer matches that chunk's own dimensions");

            Check(std::all_of(renderTasks.begin(), renderTasks.end(),
                      [](const Task& t) { return t.status == TaskStatus::COMPLETED && t.outputBlobId.has_value(); }),
                  "every Render task completed with a written output blob");
        }

        storageEngine.Close();
        projectManager.CloseProject();
        fs::remove_all(tempDir, ec);
    }

    std::vector<unsigned char> ReadFileBytes(const fs::path& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return {};
        std::streamsize size = file.tellg();
        if (size <= 0) return {};
        file.seekg(0, std::ios::beg);
        std::vector<unsigned char> bytes(static_cast<size_t>(size));
        if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) return {};
        return bytes;
    }

    // Mean squared error over interleaved RGB8 buffers, both logically
    // `width x height`, but `srcStride`/`refStride` let the caller compare
    // a sub-rectangle of a wider buffer (the exporter's composite) against
    // a same-sized reference without a separate crop/copy step.
    double ComputeMse(const unsigned char* a, int aStride, const unsigned char* b, int bStride,
                       int width, int height)
    {
        double sumSquaredError = 0.0;
        for (int y = 0; y < height; ++y)
        {
            const unsigned char* rowA = a + static_cast<size_t>(y) * aStride;
            const unsigned char* rowB = b + static_cast<size_t>(y) * bStride;
            for (int x = 0; x < width * 3; ++x)
            {
                double diff = static_cast<double>(rowA[x]) - static_cast<double>(rowB[x]);
                sumSquaredError += diff * diff;
            }
        }
        return sumSquaredError / (static_cast<double>(width) * height * 3);
    }

    double PsnrFromMse(double mse)
    {
        if (mse <= 0.0) return std::numeric_limits<double>::infinity();
        constexpr double kMaxPixelValue = 255.0;
        return 10.0 * std::log10((kMaxPixelValue * kMaxPixelValue) / mse);
    }

    void RunPanoramaQualityScenario(const fs::path& fixturesDir)
    {
        std::cout << "\n=== Scenario 3: assembled panorama vs. original scene (PSNR/MSE) ===" << std::endl;

        std::error_code ec;
        fs::path tempDir = fs::temp_directory_path() / "image360_pipeline_e2e_quality";
        fs::remove_all(tempDir, ec);
        fs::create_directories(tempDir, ec);
        Check(!ec, "create temp project directory (quality scenario)");

        ProjectManager projectManager;
        StorageEngine storageEngine;
        // Same single-chunk canvas as scenario 1, and same maxDimension as
        // totalWidth below (800) - that makes ComputeExportScale return
        // 1.0, so the export lands at native resolution with chunk
        // content placed at its exact world-space pixel offset, letting
        // us compare directly against scene_reference.jpg with no
        // resampling in the way.
        constexpr int kTotalWidth = 800;
        constexpr int kTotalHeight = 280;
        bool ranOk = SetupAndRunPipeline(tempDir, fixturesDir, kTotalWidth, kTotalHeight, 1024,
                                          projectManager, storageEngine);

        if (ranOk)
        {
            auto jpegCodec = std::make_shared<JpegCodec>();
            Check(jpegCodec->Initialize() == ComputeResult::SUCCESS, "JpegCodec::Initialize (export)");

            PanoramaExporter exporter(projectManager, storageEngine, jpegCodec);
            fs::path exportPath = tempDir / "panorama_export.jpg";
            Check(exporter.ExportPreviewJpeg(Utf8ToWide(exportPath.string()), kTotalWidth, 95),
                  "PanoramaExporter::ExportPreviewJpeg succeeds");

            auto exportedBytes = ReadFileBytes(exportPath);
            auto referenceBytes = ReadFileBytes(fixturesDir / "scene_reference.jpg");
            Check(!exportedBytes.empty(), "read back the exported panorama JPEG");
            Check(!referenceBytes.empty(), "read the scene_reference.jpg fixture");

            if (!exportedBytes.empty() && !referenceBytes.empty())
            {
                unsigned char* exportedRgb = nullptr;
                int exportedW = 0, exportedH = 0;
                unsigned char* referenceRgb = nullptr;
                int referenceW = 0, referenceH = 0;

                bool exportedDecoded = jpegCodec->Decode(exportedBytes.data(), exportedBytes.size(),
                                                          &exportedRgb, &exportedW, &exportedH) == ComputeResult::SUCCESS;
                bool referenceDecoded = jpegCodec->Decode(referenceBytes.data(), referenceBytes.size(),
                                                           &referenceRgb, &referenceW, &referenceH) == ComputeResult::SUCCESS;
                Check(exportedDecoded, "decode the exported panorama JPEG");
                Check(referenceDecoded, "decode scene_reference.jpg");

                if (exportedDecoded && referenceDecoded)
                {
                    // scene_reference.jpg is exactly the 736x240 master
                    // scene the 3 tiles were cropped from (see
                    // scripts/panorama_gen/gen_scene.py); the exported
                    // panorama is the full kTotalWidth x kTotalHeight
                    // canvas (800x280) with that same content placed at
                    // world-space offset (0,0) - compare the matching
                    // top-left sub-rectangle, not the whole canvas (the
                    // margin beyond 736x240 is background the reference
                    // image doesn't have at all).
                    Check(exportedW == kTotalWidth && exportedH == kTotalHeight,
                          "exported panorama has the expected canvas dimensions");
                    Check(referenceW == kTileWidth + 2 * 208 && referenceH == kTileHeight,
                          "scene_reference.jpg has the expected master-scene dimensions");

                    if (exportedW >= referenceW && exportedH >= referenceH)
                    {
                        double mse = ComputeMse(exportedRgb, exportedW * 3, referenceRgb, referenceW * 3,
                                                 referenceW, referenceH);
                        double psnr = PsnrFromMse(mse);
                        std::cout << "panorama vs. original scene: MSE=" << mse << " PSNR=" << psnr << " dB"
                                   << " (region " << referenceW << "x" << referenceH << ")" << std::endl;

                        // The reconstruction chain is genuinely lossy
                        // (Bayer mosaic -> demosaic -> gap-aware combine
                        // at overlap seams -> 16->8 bit -> JPEG), and the
                        // reference itself is a separately re-encoded
                        // JPEG of the same scene, so exact equality isn't
                        // expected. Empirically ~17 dB once the pipeline
                        // is actually correct (this number moved from
                        // ~6.7 dB through ~13 dB as real bugs got fixed:
                        // RenderExecutor passing the forward instead of
                        // inverse homography to WarpPerspective, the
                        // gap-vs-real-black ambiguity in combining
                        // per-chunk contributors, and DemosaicBayer's
                        // white-balance step zeroing every G2 Bayer
                        // sample when camMul[3] is unset) - a genuinely
                        // broken reconstruction (wrong homography sign,
                        // wrong chunk placement, a demosaic regression)
                        // scores far lower, typically under 10 dB in this
                        // fixture's testing. 15 dB leaves some headroom
                        // for run-to-run JPEG/demosaic noise while still
                        // catching a real regression.
                        constexpr double kMinAcceptablePsnrDb = 15.0;
                        Check(psnr >= kMinAcceptablePsnrDb,
                              "assembled panorama's PSNR against the original scene meets the quality bar");
                    }
                    else
                    {
                        Check(false, "exported panorama is at least as large as the reference region");
                    }
                }

                if (exportedRgb) jpegCodec->FreeDecoded(exportedRgb);
                if (referenceRgb) jpegCodec->FreeDecoded(referenceRgb);
            }
        }

        storageEngine.Close();
        projectManager.CloseProject();
        fs::remove_all(tempDir, ec);
    }
}

int main()
{
    fs::path fixturesDir(PIPELINE_E2E_FIXTURES_DIR);

    RunSingleChunkScenario(fixturesDir);
    RunMultiChunkScenario(fixturesDir);
    RunPanoramaQualityScenario(fixturesDir);

    if (g_failures == 0)
    {
        std::cout << "\nAll pipeline e2e checks passed." << std::endl;
        return 0;
    }
    std::cerr << "\n" << g_failures << " pipeline e2e check(s) failed." << std::endl;
    return 1;
}
