// image360-cli - headless CLI front end (docs/COMPUTATIONAL_PHOTOGRAPHY.md
// SS6, docs/superpowers/plans/2026-07-22-cli-front-end.md). A thin
// argument-parsing shell over ProjectManager/PipelineDriver/TaskScheduler,
// exposing the panorama pipeline and all four burst modes headlessly -
// mirrors how tests/pipeline_e2e*/tests/pipeline_e2e_burst already drive
// these classes in-process, just wired to real stdin args and real
// stdout/stderr instead of a fixed test scenario.
//
// Subcommands:
//   image360-cli stitch     <files...> -o out.jpg  [--chunk-size N] [--quality Q] [--backend auto|vulkan|cpu]
//   image360-cli mfnr       <files...> -o out.tif|out.jpg  [--backend ...]
//   image360-cli hdrplus    <files...> -o out.tif|out.jpg  [--backend ...]
//   image360-cli nightsight <files...> -o out.tif|out.jpg  [--backend ...]
//   image360-cli superres   <files...> -o out.tif|out.jpg  [--backend ...]
//
// Exit codes: 0 success, 1 pipeline failure, 2 CLI usage error.

#include "HeaderFiles/ProjectManager.h"
#include "HeaderFiles/StorageEngine.h"
#include "HeaderFiles/TextEncoding.h"
#include "HeaderFiles/PipelineDriver.h"
#include "HeaderFiles/RawIngestExecutor.h"
#include "HeaderFiles/AlignExecutor.h"
#include "HeaderFiles/OptimizeExecutor.h"
#include "HeaderFiles/RenderExecutor.h"
#include "HeaderFiles/PanoramaExporter.h"
#include "HeaderFiles/BurstAlignExecutor.h"
#include "HeaderFiles/BurstMergeExecutor.h"
#include "HeaderFiles/ImageLoader.h"
#include "HeaderFiles/JpegCodec.h"
#include "HeaderFiles/CpuComputeBackend.h"
#include "HeaderFiles/Tiff16Writer.h"
#include "HeaderFiles/PlatformFile.h"
#include "HeaderFiles/IComputeBackend.h"

#ifdef IMAGE360_CLI_HAVE_VULKAN
#include "HeaderFiles/VulkanPipeline.h"
#endif

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

using namespace WindowsApp::Core;
using namespace WindowsApp::Compute;
namespace fs = std::filesystem;

namespace
{
    // ---- argument parsing ---------------------------------------------

    struct CliArgs
    {
        std::string subcommand;
        std::vector<std::string> inputFiles;
        std::string outputPath;
        int quality = 90;
        int chunkSizeOverride = 0; // 0 = use the backend-derived default
        std::string backend = "auto"; // auto|vulkan|cpu
        bool valid = false;
        std::string error;
    };

    std::string ToLower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    CliArgs ParseArgs(int argc, char** argv)
    {
        CliArgs args;
        if (argc < 2)
        {
            args.error = "no subcommand given";
            return args;
        }

        args.subcommand = ToLower(argv[1]);
        static const std::vector<std::string> kKnownSubcommands = { "stitch", "mfnr", "hdrplus", "nightsight", "superres" };
        if (std::find(kKnownSubcommands.begin(), kKnownSubcommands.end(), args.subcommand) == kKnownSubcommands.end())
        {
            args.error = "unknown subcommand \"" + args.subcommand + "\"";
            return args;
        }

        for (int i = 2; i < argc; ++i)
        {
            std::string arg = argv[i];
            auto needValue = [&](const char* flagName) -> const char*
            {
                if (i + 1 >= argc) { args.error = std::string(flagName) + " needs a value"; return nullptr; }
                return argv[++i];
            };

            if (arg == "-o" || arg == "--output")
            {
                const char* v = needValue("-o/--output");
                if (!v) return args;
                args.outputPath = v;
            }
            else if (arg == "--quality")
            {
                const char* v = needValue("--quality");
                if (!v) return args;
                args.quality = std::atoi(v);
            }
            else if (arg == "--chunk-size")
            {
                const char* v = needValue("--chunk-size");
                if (!v) return args;
                args.chunkSizeOverride = std::atoi(v);
            }
            else if (arg == "--backend")
            {
                const char* v = needValue("--backend");
                if (!v) return args;
                args.backend = ToLower(v);
            }
            else if (!arg.empty() && arg[0] == '-')
            {
                args.error = "unknown flag \"" + arg + "\"";
                return args;
            }
            else
            {
                args.inputFiles.push_back(arg);
            }
        }

        if (args.inputFiles.empty()) { args.error = "no input files given"; return args; }
        if (args.outputPath.empty()) { args.error = "-o/--output is required"; return args; }
        if (args.backend != "auto" && args.backend != "vulkan" && args.backend != "cpu")
        {
            args.error = "--backend must be one of: auto, vulkan, cpu";
            return args;
        }
        if (args.quality < 1 || args.quality > 100) { args.error = "--quality must be between 1 and 100"; return args; }

        args.valid = true;
        return args;
    }

    void PrintUsage()
    {
        std::cerr <<
            "Usage: image360-cli <subcommand> <files...> -o <output> [options]\n"
            "\n"
            "Subcommands:\n"
            "  stitch      <raw-files...> -o out.jpg\n"
            "  mfnr        <burst-files...> -o out.tif|out.jpg\n"
            "  hdrplus     <burst-files...> -o out.tif|out.jpg\n"
            "  nightsight  <burst-files...> -o out.tif|out.jpg\n"
            "  superres    <burst-files...> -o out.tif|out.jpg\n"
            "\n"
            "Options:\n"
            "  -o, --output <path>   output file (required)\n"
            "  --quality <1-100>     JPEG quality, default 90\n"
            "  --chunk-size <N>      override the auto-derived render chunk size (stitch only)\n"
            "  --backend auto|vulkan|cpu   compute backend, default auto (CUDA not yet wired\n"
            "                              into this CLI's build - see docs/superpowers/plans/\n"
            "                              2026-07-22-cli-front-end.md scope decision 3)\n";
    }

    // ---- CfaType detection (mirrors WindowsApp/MainWindow.xaml.cpp's own logic) ----

    struct DetectedImage
    {
        std::wstring path;
        CfaType cfaType = CfaType::UNKNOWN;
        int width = 0;
        int height = 0;
    };

    bool DetectImage(const std::wstring& path, DetectedImage& out)
    {
        out.path = path;

        if (IsJpegFile(path) || IsStandardImageFile(path))
        {
            if (!GetStandardImageDimensions(path, out.width, out.height)) return false;
            if (out.width <= 0 || out.height <= 0) return false;
            out.cfaType = CfaType::STANDARD_RGB;
            return true;
        }

        ImageLoader loader;
        if (!loader.Open(path)) return false;

        RawPlane plane;
        if (loader.UnpackRaw(plane))
        {
            out.cfaType = plane.cfaType;
            out.width = plane.width;
            out.height = plane.height;
        }
        else
        {
            ImageMetadata metadata;
            if (!loader.GetMetadata(metadata)) return false;
            out.width = metadata.width;
            out.height = metadata.height;
        }

        return out.width > 0 && out.height > 0;
    }

    // ---- compute backend selection (CLI-local; see plan doc scope decision 3) ----

    struct SelectedBackend
    {
        std::shared_ptr<IComputeBackend> backend;
        std::shared_ptr<IImageCodec> jpegCodec;
        bool usedGpu = false;
        std::string name;
    };

    bool TryVulkan(SelectedBackend& sel)
    {
#ifdef IMAGE360_CLI_HAVE_VULKAN
        auto vulkan = std::make_shared<VulkanPipeline>();
        if (vulkan->Initialize() != ComputeResult::SUCCESS) return false;

        auto jpeg = std::make_shared<JpegCodec>();
        if (jpeg->Initialize() != ComputeResult::SUCCESS) return false;

        sel.backend = vulkan;
        sel.jpegCodec = jpeg;
        sel.usedGpu = true;
        sel.name = std::string("Vulkan (") + vulkan->GetGpuInfo().name + ")";
        return true;
#else
        (void)sel;
        return false;
#endif
    }

    // allowGpu: BlockMatchAlign/RobustMergeAccumulate/TileFftMerge/
    // StructureTensorKernelRegression (every burst-mode op) are CPU-only -
    // CudaPipeline/VulkanPipeline only have typed NOT_SUPPORTED stubs for
    // them (a real, documented, tracked gap since Phase 1, not a bug) -
    // so RunBurst always passes allowGpu=false regardless of --backend,
    // rather than letting a GPU-capable machine silently fail every
    // BurstAlign task. RunStitch passes true - the panorama executors are
    // real on all three tiers.
    SelectedBackend SelectBackend(const std::string& preferred, bool allowGpu)
    {
        SelectedBackend sel;

        if (allowGpu && (preferred == "auto" || preferred == "vulkan"))
        {
            if (TryVulkan(sel)) return sel;
            if (preferred == "vulkan")
                std::cerr << "warning: --backend vulkan requested but unavailable, falling back to CPU" << std::endl;
        }
        else if (!allowGpu && preferred != "auto" && preferred != "cpu")
        {
            std::cerr << "warning: --backend " << preferred << " ignored - burst-mode kernels are CPU-only "
                          "(BlockMatchAlign and friends have no Vulkan/CUDA implementation yet), using CPU"
                       << std::endl;
        }

        auto cpu = std::make_shared<CpuComputeBackend>();
        cpu->Initialize(); // cannot fail - always some SIMD tier available
        auto cpuJpeg = std::make_shared<JpegCodec>();
        cpuJpeg->Initialize();

        sel.backend = cpu;
        sel.jpegCodec = cpuJpeg;
        sel.usedGpu = false;
        sel.name = std::string("CPU (") + cpu->GetGpuInfo().name + ")";
        return sel;
    }

    int RecommendedChunkSizeFor(const SelectedBackend& sel, int override_)
    {
        if (override_ > 0) return override_;
        uint64_t totalMemory = sel.backend->GetGpuInfo().totalMemory;
        return sel.usedGpu ? RecommendedChunkSize(totalMemory) : RecommendedChunkSizeForCpu(totalMemory);
    }

    // ---- output writing (burst modes) ----

    bool WriteBurstOutput(const std::string& outputPath, const PixelBuffer& buffer,
                           const std::shared_ptr<IImageCodec>& jpegCodec, int quality)
    {
        std::string ext = ToLower(fs::path(outputPath).extension().string());
        std::wstring wOutputPath = Utf8ToWide(outputPath);

        if (ext == ".tif" || ext == ".tiff")
        {
            return WriteTiff16RGB(wOutputPath, buffer.data.data(), buffer.width, buffer.height);
        }
        if (ext == ".jpg" || ext == ".jpeg")
        {
            std::vector<unsigned char> rgb8(buffer.data.size());
            for (size_t i = 0; i < buffer.data.size(); ++i) rgb8[i] = static_cast<unsigned char>(buffer.data[i] >> 8);

            std::vector<unsigned char> jpegBytes;
            if (jpegCodec->Encode(rgb8.data(), buffer.width, buffer.height, quality, jpegBytes) != ComputeResult::SUCCESS)
                return false;

            PlatformFile file;
            if (!file.Open(wOutputPath, FileOpenMode::CreateAlways)) return false;
            bool ok = file.Write(jpegBytes.data(), jpegBytes.size());
            file.Close();
            return ok;
        }

        std::cerr << "error: unsupported output extension \"" << ext << "\" - use .tif/.tiff or .jpg/.jpeg" << std::endl;
        return false;
    }

    // ---- progress/log callbacks ----

    PipelineDriver::ProgressCallback MakeProgressCallback()
    {
        return [](PipelineStage stage, float progress)
        {
            std::cout << "[stage " << static_cast<int>(stage) << "] " << static_cast<int>(progress * 100.0f) << "%"
                       << std::endl;
        };
    }

    PipelineDriver::LogCallback MakeLogCallback()
    {
        return [](const std::wstring& msg) { std::wcerr << L"log: " << msg << std::endl; };
    }

    size_t MaxInFlightFor(const SelectedBackend& sel)
    {
        if (sel.usedGpu) return 2;
        return (std::max)(size_t(1), static_cast<size_t>(std::thread::hardware_concurrency()) - 1);
    }

    // ---- stitch ----

    int RunStitch(const CliArgs& args)
    {
        std::string ext = ToLower(fs::path(args.outputPath).extension().string());
        if (ext != ".jpg" && ext != ".jpeg")
        {
            std::cerr << "error: stitch's -o must end in .jpg or .jpeg (PanoramaExporter only writes JPEG)"
                       << std::endl;
            return 2;
        }

        std::vector<DetectedImage> images;
        for (const auto& f : args.inputFiles)
        {
            DetectedImage img;
            if (!DetectImage(Utf8ToWide(f), img))
            {
                std::cerr << "error: failed to open/detect \"" << f << "\"" << std::endl;
                return 1;
            }
            images.push_back(img);
        }

        // Coarse pre-alignment canvas over-estimate - same heuristic
        // WindowsApp/MainWindow.xaml.cpp already uses (docs/superpowers/
        // plans/2026-07-22-cli-front-end.md scope decision 2).
        int maxWidth = 0, maxHeight = 0;
        for (const auto& img : images)
        {
            maxWidth = (std::max)(maxWidth, img.width);
            maxHeight = (std::max)(maxHeight, img.height);
        }
        int totalWidth = maxWidth * static_cast<int>(images.size());
        int totalHeight = maxHeight * static_cast<int>(images.size());

        SelectedBackend sel = SelectBackend(args.backend, /*allowGpu=*/true);
        std::cout << "compute backend: " << sel.name << std::endl;
        int chunkSize = RecommendedChunkSizeFor(sel, args.chunkSizeOverride);

        std::error_code ec;
        fs::path tempDir = fs::temp_directory_path() / "image360_cli_stitch";
        fs::remove_all(tempDir, ec);
        fs::create_directories(tempDir, ec);
        if (ec) { std::cerr << "error: failed to create temp project directory" << std::endl; return 1; }

        ProjectManager projectManager;
        fs::path dbPath = tempDir / "project.vfp";
        if (!projectManager.CreateProject(Utf8ToWide(dbPath.string()), totalWidth, totalHeight, chunkSize))
        {
            std::cerr << "error: ProjectManager::CreateProject failed" << std::endl;
            fs::remove_all(tempDir, ec);
            return 1;
        }

        for (const auto& img : images)
        {
            if (!projectManager.AddInputImage(img.path, Homography{}, img.cfaType))
            {
                std::cerr << "error: failed to register input image" << std::endl;
                fs::remove_all(tempDir, ec);
                return 1;
            }
        }

        StorageEngine storageEngine;
        if (!storageEngine.Open(Utf8ToWide(tempDir.string()), L"project", projectManager))
        {
            std::cerr << "error: StorageEngine::Open failed" << std::endl;
            fs::remove_all(tempDir, ec);
            return 1;
        }

        bool seeded = projectManager.SeedIngestTasks() && projectManager.SeedAlignTasks()
            && projectManager.SeedOptimizeTasks();
        if (!seeded)
        {
            std::cerr << "error: failed to seed pipeline tasks" << std::endl;
            storageEngine.Close();
            projectManager.CloseProject();
            fs::remove_all(tempDir, ec);
            return 1;
        }

        PipelineDriver driver;
        driver.Initialize(MakeProgressCallback(), MakeLogCallback(), MaxInFlightFor(sel));
        driver.RegisterExecutor(PipelineStage::STAGE0_INGEST,
            std::make_shared<RawIngestExecutor>(projectManager, storageEngine, sel.backend, sel.jpegCodec));
        driver.RegisterExecutor(PipelineStage::STAGE1_ALIGN,
            std::make_shared<AlignExecutor>(projectManager, storageEngine, sel.backend, sel.jpegCodec));
        driver.RegisterExecutor(PipelineStage::STAGE2_OPTIMIZE,
            std::make_shared<OptimizeExecutor>(projectManager, storageEngine, sel.backend, sel.jpegCodec));
        driver.RegisterExecutor(PipelineStage::STAGE3_RENDER,
            std::make_shared<RenderExecutor>(projectManager, storageEngine, sel.backend));

        std::stop_source stopSource;
        bool ranOk = driver.Run(projectManager, stopSource.get_token());
        int result = 0;
        if (!ranOk || driver.GetCurrentStage() != PipelineStage::COMPLETED)
        {
            std::cerr << "error: pipeline did not complete successfully" << std::endl;
            result = 1;
        }
        else
        {
            PanoramaExporter exporter(projectManager, storageEngine, sel.jpegCodec);
            if (!exporter.ExportPreviewJpeg(Utf8ToWide(args.outputPath), totalWidth, args.quality))
            {
                std::cerr << "error: failed to export panorama" << std::endl;
                result = 1;
            }
            else
            {
                std::cout << "wrote " << args.outputPath << std::endl;
            }
        }

        storageEngine.Close();
        projectManager.CloseProject();
        fs::remove_all(tempDir, ec);
        return result;
    }

    // ---- burst modes (mfnr/hdrplus/nightsight/superres) ----

    int RunBurst(const CliArgs& args, BurstMode mode)
    {
        std::string ext = ToLower(fs::path(args.outputPath).extension().string());
        if (ext != ".tif" && ext != ".tiff" && ext != ".jpg" && ext != ".jpeg")
        {
            std::cerr << "error: -o must end in .tif/.tiff (lossless) or .jpg/.jpeg (lossy)" << std::endl;
            return 2;
        }
        if (args.inputFiles.size() < 2)
        {
            std::cerr << "error: burst modes need at least 2 input frames" << std::endl;
            return 2;
        }

        std::vector<DetectedImage> images;
        for (const auto& f : args.inputFiles)
        {
            DetectedImage img;
            if (!DetectImage(Utf8ToWide(f), img))
            {
                std::cerr << "error: failed to open/detect \"" << f << "\"" << std::endl;
                return 1;
            }
            images.push_back(img);
        }

        SelectedBackend sel = SelectBackend(args.backend, /*allowGpu=*/false);
        std::cout << "compute backend: " << sel.name << std::endl;

        std::error_code ec;
        fs::path tempDir = fs::temp_directory_path() / "image360_cli_burst";
        fs::remove_all(tempDir, ec);
        fs::create_directories(tempDir, ec);
        if (ec) { std::cerr << "error: failed to create temp project directory" << std::endl; return 1; }

        ProjectManager projectManager;
        fs::path dbPath = tempDir / "project.vfp";
        if (!projectManager.CreateBurstProject(Utf8ToWide(dbPath.string()), mode))
        {
            std::cerr << "error: ProjectManager::CreateBurstProject failed" << std::endl;
            fs::remove_all(tempDir, ec);
            return 1;
        }

        for (const auto& img : images)
        {
            if (!projectManager.AddInputImage(img.path, Homography{}, img.cfaType))
            {
                std::cerr << "error: failed to register input image" << std::endl;
                fs::remove_all(tempDir, ec);
                return 1;
            }
        }

        StorageEngine storageEngine;
        if (!storageEngine.Open(Utf8ToWide(tempDir.string()), L"project", projectManager))
        {
            std::cerr << "error: StorageEngine::Open failed" << std::endl;
            fs::remove_all(tempDir, ec);
            return 1;
        }

        if (!projectManager.SeedBurstAlignTasks() || !projectManager.SeedBurstMergeTasks())
        {
            std::cerr << "error: failed to seed pipeline tasks" << std::endl;
            storageEngine.Close();
            projectManager.CloseProject();
            fs::remove_all(tempDir, ec);
            return 1;
        }

        auto jpegCodecForAlign = std::make_shared<JpegCodec>();
        jpegCodecForAlign->Initialize();

        PipelineDriver driver;
        driver.Initialize(MakeProgressCallback(), MakeLogCallback(), MaxInFlightFor(sel));
        driver.RegisterExecutor(PipelineStage::BURST_ALIGN,
            std::make_shared<BurstAlignExecutor>(projectManager, storageEngine, sel.backend, jpegCodecForAlign));
        auto mergeExecutor = std::make_shared<BurstMergeExecutor>(projectManager, storageEngine, sel.backend);
        driver.RegisterExecutor(PipelineStage::BURST_MERGE, mergeExecutor);
        driver.RegisterExecutor(PipelineStage::BURST_FINISH, mergeExecutor);

        std::stop_source stopSource;
        bool ranOk = driver.Run(projectManager, stopSource.get_token());
        int result = 0;
        if (!ranOk || driver.GetCurrentStage() != PipelineStage::COMPLETED)
        {
            std::cerr << "error: pipeline did not complete successfully" << std::endl;
            result = 1;
        }
        else
        {
            std::vector<Task> finishTasks = projectManager.GetTasksForStage(PipelineStage::BURST_FINISH);
            std::optional<PixelBuffer> finalBuffer;
            if (finishTasks.size() == 1 && finishTasks[0].status == TaskStatus::COMPLETED
                && finishTasks[0].outputBlobId.has_value())
            {
                finalBuffer = storageEngine.ReadPixelBuffer(*finishTasks[0].outputBlobId);
            }

            if (!finalBuffer.has_value())
            {
                std::cerr << "error: failed to read BURST_FINISH's output" << std::endl;
                result = 1;
            }
            else if (!WriteBurstOutput(args.outputPath, *finalBuffer, sel.jpegCodec, args.quality))
            {
                std::cerr << "error: failed to write output file" << std::endl;
                result = 1;
            }
            else
            {
                std::cout << "wrote " << args.outputPath << " (" << finalBuffer->width << "x" << finalBuffer->height
                           << ")" << std::endl;
            }
        }

        storageEngine.Close();
        projectManager.CloseProject();
        fs::remove_all(tempDir, ec);
        return result;
    }
}

int main(int argc, char** argv)
{
    CliArgs args = ParseArgs(argc, argv);
    if (!args.valid)
    {
        std::cerr << "error: " << args.error << "\n\n";
        PrintUsage();
        return 2;
    }

    if (args.subcommand == "stitch") return RunStitch(args);
    if (args.subcommand == "mfnr") return RunBurst(args, BurstMode::MFNR);
    if (args.subcommand == "hdrplus") return RunBurst(args, BurstMode::HDR_PLUS);
    if (args.subcommand == "nightsight") return RunBurst(args, BurstMode::NIGHT_SIGHT);
    if (args.subcommand == "superres") return RunBurst(args, BurstMode::SUPER_RES);

    PrintUsage();
    return 2;
}
