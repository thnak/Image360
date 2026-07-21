// Full pipeline integration test using REAL phone-camera JPEGs (not the
// synthetic hand-rolled DNG tiles tests/pipeline_e2e uses). Exercises the
// CfaType::STANDARD_RGB path end-to-end - RawIngestExecutor's JPEG
// decode, AlignExecutor::ExecuteFeatureExtraction's/OptimizeExecutor::
// ExecuteGain's preview decode, and OverlapCulling::FindOverlappingImages'
// dimension probe - against real-world files. This is exactly what caught
// several RAW-only assumptions baked into those executors while adding
// JPEG/PNG input support (each one independently re-opened the original
// file via ImageLoader/LibRaw for its own purposes, and none of them
// handled a plain consumer image format).
//
// Unlike tests/pipeline_e2e, there's no known-correct geometry for real
// photos to check against - this is a pass/fail smoke test: does the
// whole pipeline (Ingest -> Align -> Optimize -> Render) complete with
// every task COMPLETED, not "did it recover the right homography".
//
// CI-only (see IMAGE360_ENABLE_REAL_PHOTO_TESTS in the root CMakeLists.txt)
// - not registered by a default local build.

#include "HeaderFiles/ProjectManager.h"
#include "HeaderFiles/StorageEngine.h"
#include "HeaderFiles/TextEncoding.h"
#include "HeaderFiles/PipelineDriver.h"
#include "HeaderFiles/RawIngestExecutor.h"
#include "HeaderFiles/AlignExecutor.h"
#include "HeaderFiles/OptimizeExecutor.h"
#include "HeaderFiles/RenderExecutor.h"
#include "HeaderFiles/CpuComputeBackend.h"
#include "HeaderFiles/JpegCodec.h"
#include "HeaderFiles/ImageLoader.h"

#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <iostream>
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

    namespace fs = std::filesystem;
}

int main()
{
    fs::path fixturesDir(REAL_PHOTOS_FIXTURES_DIR);

    std::vector<fs::path> photoPaths;
    for (const auto& entry : fs::directory_iterator(fixturesDir))
    {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg")
        {
            photoPaths.push_back(entry.path());
        }
    }
    std::sort(photoPaths.begin(), photoPaths.end());

    Check(photoPaths.size() >= 2, "at least 2 real photo fixtures found");
    if (photoPaths.size() < 2)
    {
        std::cerr << g_failures << " check(s) failed." << std::endl;
        return 1;
    }

    std::error_code ec;
    fs::path tempDir = fs::temp_directory_path() / "image360_real_photos_e2e";
    fs::remove_all(tempDir, ec);
    fs::create_directories(tempDir, ec);
    Check(!ec, "create temp project directory");

    // Coarse canvas estimate, matching MainWindow::StitchStartButton_Click's
    // own pre-alignment formula - real photos have no known tile geometry
    // to size the canvas from precisely.
    int maxWidth = 0, maxHeight = 0;
    for (const auto& path : photoPaths)
    {
        int w = 0, h = 0;
        Check(GetStandardImageDimensions(path.wstring(), w, h), "GetStandardImageDimensions succeeds for a real photo");
        maxWidth = (std::max)(maxWidth, w);
        maxHeight = (std::max)(maxHeight, h);
    }
    int totalWidth = maxWidth * static_cast<int>(photoPaths.size());
    int totalHeight = maxHeight * static_cast<int>(photoPaths.size());

    ProjectManager projectManager;
    StorageEngine storageEngine;

    fs::path dbPath = tempDir / "real_photos.vfp";
    Check(projectManager.CreateProject(Utf8ToWide(dbPath.string()), totalWidth, totalHeight, 4096),
          "ProjectManager::CreateProject");

    for (const auto& path : photoPaths)
    {
        Check(projectManager.AddInputImage(Utf8ToWide(path.string()), Homography{}, CfaType::STANDARD_RGB),
              "ProjectManager::AddInputImage (real photo)");
    }
    Check(projectManager.GetInputImages().size() == photoPaths.size(), "all real photos registered");

    Check(storageEngine.Open(Utf8ToWide(tempDir.string()), L"real_photos", projectManager), "StorageEngine::Open");

    Check(projectManager.SeedIngestTasks(), "ProjectManager::SeedIngestTasks");
    Check(projectManager.SeedAlignTasks(), "ProjectManager::SeedAlignTasks");
    Check(projectManager.SeedOptimizeTasks(), "ProjectManager::SeedOptimizeTasks");

    auto computeBackend = std::make_shared<CpuComputeBackend>();
    Check(computeBackend->Initialize() == ComputeResult::SUCCESS, "CpuComputeBackend::Initialize");

    auto jpegCodec = std::make_shared<JpegCodec>();
    Check(jpegCodec->Initialize() == ComputeResult::SUCCESS, "JpegCodec::Initialize");

    PipelineDriver driver;
    size_t maxInFlight = (std::max)(size_t(1), static_cast<size_t>(std::thread::hardware_concurrency()) - 1);
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
        std::make_shared<RawIngestExecutor>(projectManager, storageEngine, computeBackend, jpegCodec));
    driver.RegisterExecutor(PipelineStage::STAGE1_ALIGN,
        std::make_shared<AlignExecutor>(projectManager, storageEngine, computeBackend, jpegCodec));
    driver.RegisterExecutor(PipelineStage::STAGE2_OPTIMIZE,
        std::make_shared<OptimizeExecutor>(projectManager, storageEngine, computeBackend, jpegCodec));
    driver.RegisterExecutor(PipelineStage::STAGE3_RENDER,
        std::make_shared<RenderExecutor>(projectManager, storageEngine, computeBackend));

    std::stop_source stopSource;
    bool ranOk = driver.Run(projectManager, stopSource.get_token());
    Check(ranOk, "PipelineDriver::Run completes Ingest->Align->Optimize->Render on real photos");
    Check(driver.GetCurrentStage() == PipelineStage::COMPLETED, "PipelineDriver ends in COMPLETED stage");

    // The specific regression class this test exists to catch: every
    // stage's tasks must be COMPLETED, not silently failed/skipped
    // because some executor assumed a RAW-only input.
    for (PipelineStage stage : { PipelineStage::STAGE0_INGEST, PipelineStage::STAGE1_ALIGN,
                                 PipelineStage::STAGE2_OPTIMIZE, PipelineStage::STAGE3_RENDER })
    {
        std::vector<Task> tasks = projectManager.GetTasksForStage(stage);
        bool allCompleted = !tasks.empty() && std::all_of(tasks.begin(), tasks.end(),
            [](const Task& t) { return t.status == TaskStatus::COMPLETED; });
        Check(allCompleted, "every task in this stage completed for the real-photo project");
    }

    storageEngine.Close();
    projectManager.CloseProject();
    fs::remove_all(tempDir, ec);

    if (g_failures == 0)
    {
        std::cout << "\nAll real-photo pipeline e2e checks passed." << std::endl;
        return 0;
    }
    std::cerr << "\n" << g_failures << " real-photo pipeline e2e check(s) failed." << std::endl;
    return 1;
}
