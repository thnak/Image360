#include "pch.h"
#include "MainWindow.xaml.h"
#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif

#include "ImageLoader.h"
#include "PanoramaExporter.h"
#include "ComputeBackendFactory.h"
#include "CpuComputeBackend.h"
#include "JpegCodec.h"
#include "BurstAlignExecutor.h"
#include "BurstMergeExecutor.h"
#include "Tiff16Writer.h"
#include "PlatformFile.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <cwctype>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace winrt;
using namespace Microsoft::UI::Xaml;
using namespace Microsoft::UI::Xaml::Media::Imaging;
using namespace Microsoft::Windows::Storage::Pickers;
using namespace Windows::Storage;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace
{
    winrt::hstring StageToDisplayString(::WindowsApp::Core::PipelineStage stage)
    {
        using ::WindowsApp::Core::PipelineStage;
        switch (stage)
        {
        case PipelineStage::IDLE:            return L"Idle.";
        case PipelineStage::STAGE0_INGEST:   return L"Ingesting raw images...";
        case PipelineStage::STAGE1_ALIGN:    return L"Aligning images...";
        case PipelineStage::STAGE2_OPTIMIZE: return L"Optimizing exposure & geometry...";
        case PipelineStage::STAGE3_RENDER:   return L"Rendering panorama...";
        case PipelineStage::BURST_ALIGN:     return L"Aligning burst frames...";
        case PipelineStage::BURST_MERGE:     return L"Merging burst frames...";
        case PipelineStage::BURST_FINISH:    return L"Finishing...";
        case PipelineStage::COMPLETED:       return L"Run completed.";
        case PipelineStage::CANCELLED:       return L"Run cancelled.";
        case PipelineStage::FAILED:          return L"Run failed.";
        }
        return L"Running...";
    }

    // ApplicationData::Current() throws ("The process has no package
    // identity") unless the app is running with MSIX package identity -
    // i.e. launched via the WindowsApp (Package) project, not the bare
    // WindowsApp.exe. Fall back to %LOCALAPPDATA%\Image360 so the demo
    // run works either way instead of crashing when run unpackaged. One
    // project file reused/overwritten per run regardless of which
    // feature is selected - it's scratch state, not something a user
    // needs to keep across runs (docs/superpowers/plans/
    // 2026-07-22-winui-feature-management.md scope decision 4's renamed
    // "Stitch*" -> feature-neutral naming applies here too).
    std::wstring RunProjectPath()
    {
        try
        {
            winrt::hstring localFolder = ApplicationData::Current().LocalFolder().Path();
            return std::wstring(localFolder.c_str()) + L"\\demo_run.vfp";
        }
        catch (hresult_error const&)
        {
            wchar_t buffer[MAX_PATH];
            DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", buffer, MAX_PATH);
            std::wstring folder = (len > 0 && len < MAX_PATH) ? std::wstring(buffer) : L".";
            folder += L"\\Image360";
            CreateDirectoryW(folder.c_str(), nullptr);
            return folder + L"\\demo_run.vfp";
        }
    }

    // Same file/format App::App()'s UnhandledException handler writes to -
    // one place to check regardless of whether a run failed gracefully
    // (this) or crashed outright (that). Pipeline log lines only ever
    // reach the UI as a transient RunStatusText string that the
    // completion handler immediately overwrites with "Run failed.", so
    // without this the reason is gone by the time the user can read it.
    void AppendToStitchLog(const std::wstring& message)
    {
        wchar_t tempPath[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, tempPath)) return;

        std::wstring logFilePath = std::wstring(tempPath) + L"WindowsApp.log";
        std::wofstream logFile(logFilePath, std::ios::app);
        if (!logFile.is_open()) return;

        time_t now = time(nullptr);
        wchar_t timeStr[26];
        _wctime_s(timeStr, 26, &now);
        std::wstring timeString(timeStr);
        if (!timeString.empty() && timeString.back() == L'\n')
        {
            timeString.pop_back();
        }
        logFile << L"[" << timeString << L"] " << message << std::endl;
    }

    using ::winrt::WindowsApp::implementation::AppFeature;

    bool IsBurstFeature(AppFeature feature)
    {
        return feature != AppFeature::Stitch;
    }

    ::WindowsApp::Core::BurstMode ToBurstMode(AppFeature feature)
    {
        using ::WindowsApp::Core::BurstMode;
        switch (feature)
        {
        case AppFeature::Mfnr:       return BurstMode::MFNR;
        case AppFeature::HdrPlus:    return BurstMode::HDR_PLUS;
        case AppFeature::NightSight: return BurstMode::NIGHT_SIGHT;
        case AppFeature::SuperRes:   return BurstMode::SUPER_RES;
        default:                     return BurstMode::NONE; // unreachable for Stitch callers
        }
    }

    struct FeatureChrome
    {
        const wchar_t* headerTitle;
        const wchar_t* headerSubtitle;
        const wchar_t* cardTitle;
        const wchar_t* startLabel;
    };

    // Text shown for whichever feature FeatureComboBox has selected -
    // see MainWindow::FeatureComboBox_SelectionChanged.
    FeatureChrome ChromeFor(AppFeature feature)
    {
        switch (feature)
        {
        case AppFeature::Mfnr:
            return { L"MFNR", L"Pick a burst of frames from the same scene to denoise via multi-frame merge.",
                     L"MFNR (Noise Reduction)", L"Start MFNR" };
        case AppFeature::HdrPlus:
            return { L"HDR+", L"Pick a burst of frames from the same scene to merge into one high-dynamic-range image.",
                     L"HDR+", L"Start HDR+" };
        case AppFeature::NightSight:
            return { L"Night Sight", L"Pick a handheld low-light burst to denoise and tone-map.",
                     L"Night Sight", L"Start Night Sight" };
        case AppFeature::SuperRes:
            return { L"Super Res Zoom", L"Pick a burst of frames to reconstruct a higher-resolution image from sub-pixel detail.",
                     L"Super Res Zoom", L"Start Super Res Zoom" };
        default:
            return { L"Panorama Stitcher", L"Pick a set of overlapping RAW photos and stitch them into one panorama.",
                     L"Panorama Stitch", L"Start Stitch" };
        }
    }
}

namespace winrt::WindowsApp::implementation
{
    int32_t MainWindow::MyProperty()
    {
        throw hresult_not_implemented();
    }

    void MainWindow::MyProperty(int32_t /* value */)
    {
        throw hresult_not_implemented();
    }

    void MainWindow::FeatureComboBox_SelectionChanged(
        Windows::Foundation::IInspectable const& /* sender */,
        Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& /* args */)
    {
        switch (FeatureComboBox().SelectedIndex())
        {
        case 1: m_selectedFeature = AppFeature::Mfnr; break;
        case 2: m_selectedFeature = AppFeature::HdrPlus; break;
        case 3: m_selectedFeature = AppFeature::NightSight; break;
        case 4: m_selectedFeature = AppFeature::SuperRes; break;
        default: m_selectedFeature = AppFeature::Stitch; break;
        }

        FeatureChrome chrome = ChromeFor(m_selectedFeature);
        HeaderTitleText().Text(chrome.headerTitle);
        HeaderSubtitleText().Text(chrome.headerSubtitle);
        FeatureCardTitle().Text(chrome.cardTitle);
        RunStartButton().Content(winrt::box_value(winrt::hstring{ chrome.startLabel }));

        // Whether the cached compute backend needs to be re-selected is
        // decided in RunStartButton_Click by comparing
        // m_lastBackendForcedForBurst against the newly selected
        // feature's own requirement (burst modes always force CPU - see
        // the plan doc's scope decision 1) - no invalidation needed here.
    }

    winrt::fire_and_forget MainWindow::RunStartButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        using ::WindowsApp::Core::CfaType;
        using ::WindowsApp::Core::Homography;
        using ::WindowsApp::Core::ImageMetadata;
        using ::WindowsApp::Core::PipelineStage;
        using ::WindowsApp::Core::RawPlane;
        using ::WindowsApp::Core::Task;
        using ::WindowsApp::Core::TaskStatus;

        auto lifetime{ get_strong() }; // keep `this` alive across co_await suspension

        bool isBurst = IsBurstFeature(m_selectedFeature);

        RunStartButton().IsEnabled(false);
        RunCancelButton().IsEnabled(true);
        RunStatusText().Text(L"Pick photos to process...");
        RunProgressBar().Value(0);
        m_lastLogMessage.clear();

        // A re-run (Start clicked again after a previous run) must not
        // keep showing the previous run's finished canvas while photos
        // are (re-)picked - back to the empty state until a new result
        // exists.
        EmptyStatePanel().Visibility(Visibility::Visible);
        RenderPreviewImage().Source(nullptr);
        m_renderCanvas = nullptr;

        // A previous run has already finished (Start is disabled while
        // one is active), but std::jthread stays joinable until
        // explicitly joined - clear it out before starting a fresh one.
        if (m_runThread.joinable())
        {
            m_runThread.join();
        }

        // Lazily (re-)initialize the engine objects. Burst modes always
        // force CPU (BlockMatchAlign/RobustMergeAccumulate/TileFftMerge/
        // StructureTensorKernelRegression have no Vulkan/CUDA
        // implementation yet - see the plan doc's scope decision 1);
        // Stitch keeps SelectComputeBackend()'s existing Auto/CUDA/
        // Vulkan/CPU behavior. Re-selects whenever the cached backend's
        // forced-CPU-or-not policy doesn't match what the newly selected
        // feature needs, not just once per session, so switching between
        // Stitch and a burst mode (or back) always picks correctly.
        if (!m_computeInitialized || m_lastBackendForcedForBurst != isBurst)
        {
            if (isBurst)
            {
                auto cpu = std::make_shared<::WindowsApp::Core::CpuComputeBackend>();
                cpu->Initialize(); // cannot fail - always some SIMD tier available
                auto cpuJpeg = std::make_shared<::WindowsApp::Core::JpegCodec>();
                cpuJpeg->Initialize();

                auto cpuInfo = cpu->GetGpuInfo();
                std::wstring cpuName(cpuInfo.name, cpuInfo.name + strlen(cpuInfo.name));

                m_cudaPipeline = cpu;
                m_nvJpegCodec = cpuJpeg;
                m_computeMaxInFlight = (std::max)(size_t(1),
                    static_cast<size_t>(std::thread::hardware_concurrency()) - 1);
                AppendToStitchLog(
                    L"Burst modes run on CPU only (no Vulkan/CUDA implementation yet for "
                    L"BlockMatchAlign and friends) - using " + cpuName);
            }
            else
            {
                auto selection = ::WindowsApp::SelectComputeBackend(m_computeBackendPreference);
                m_cudaPipeline = selection.backend;
                m_nvJpegCodec = selection.codec;
                m_computeMaxInFlight = selection.recommendedMaxInFlight;
                AppendToStitchLog(selection.statusMessage);
            }

            m_lastBackendForcedForBurst = isBurst;
            m_computeInitialized = true;
        }

        FileOpenPicker picker{ AppWindow().Id() };
        picker.FileTypeFilter().Append(L".raf");
        picker.FileTypeFilter().Append(L".cr2");
        picker.FileTypeFilter().Append(L".nef");
        picker.FileTypeFilter().Append(L".arw");
        picker.FileTypeFilter().Append(L".dng");
        // Standard consumer image formats - what most phone cameras
        // actually produce, routed around ImageLoader/LibRaw entirely
        // (see RawIngestExecutor::Execute's STANDARD_RGB branch).
        picker.FileTypeFilter().Append(L".jpg");
        picker.FileTypeFilter().Append(L".jpeg");
        picker.FileTypeFilter().Append(L".png");
        picker.FileTypeFilter().Append(L".bmp");
        picker.FileTypeFilter().Append(L".gif");
        picker.FileTypeFilter().Append(L".tga");
        picker.FileTypeFilter().Append(L".tif");
        picker.FileTypeFilter().Append(L".tiff");

        auto pickedFiles{ co_await picker.PickMultipleFilesAsync() };
        if (!pickedFiles || pickedFiles.Size() == 0)
        {
            RunStartButton().IsEnabled(true);
            RunCancelButton().IsEnabled(false);
            RunStatusText().Text(L"No photos selected.");
            co_return;
        }

        if (isBurst && pickedFiles.Size() < 2)
        {
            RunStartButton().IsEnabled(true);
            RunCancelButton().IsEnabled(false);
            RunStatusText().Text(L"Burst modes need at least 2 frames.");
            co_return;
        }

        RunStatusText().Text(L"Reading photo metadata...");

        // Per-file LibRaw metadata/unpack reads run synchronously here on
        // whatever thread resumes this coroutine (typically the UI
        // thread) - a known limitation for large multi-file picks, not
        // solved in this pass; the pipeline *run* itself (below) is what
        // this plan's non-blocking guarantee actually covers.
        struct PickedImage
        {
            std::wstring path;
            CfaType cfaType;
            int width;
            int height;
        };

        std::vector<PickedImage> pickedImages;
        for (auto const& file : pickedFiles)
        {
            std::wstring path(file.Path().c_str());

            if (::WindowsApp::Core::IsJpegFile(path) || ::WindowsApp::Core::IsStandardImageFile(path))
            {
                // Header-only read (no full decode) just for width/height
                // (the pre-alignment canvas estimate below) -
                // RawIngestExecutor::Execute's STANDARD_RGB branch decodes
                // the actual pixels again during Stage0, same
                // throwaway-metadata-read pattern the RAW branch below
                // already has with its own UnpackRaw/GetMetadata call.
                int standardWidth = 0, standardHeight = 0;
                if (!::WindowsApp::Core::GetStandardImageDimensions(path, standardWidth, standardHeight)) continue;
                if (standardWidth <= 0 || standardHeight <= 0) continue;

                pickedImages.push_back({ path, CfaType::STANDARD_RGB, standardWidth, standardHeight });
                continue;
            }

            ::WindowsApp::Core::ImageLoader loader;
            if (!loader.Open(path)) continue;

            CfaType cfaType = CfaType::UNKNOWN;
            int width = 0;
            int height = 0;

            RawPlane plane;
            if (loader.UnpackRaw(plane))
            {
                cfaType = plane.cfaType;
                width = plane.width;
                height = plane.height;
            }
            else
            {
                ImageMetadata metadata;
                if (loader.GetMetadata(metadata))
                {
                    width = metadata.width;
                    height = metadata.height;
                }
            }

            if (width <= 0 || height <= 0) continue;

            pickedImages.push_back({ path, cfaType, width, height });
        }

        if (pickedImages.empty() || (isBurst && pickedImages.size() < 2))
        {
            RunStartButton().IsEnabled(true);
            RunCancelButton().IsEnabled(false);
            RunStatusText().Text(L"Could not read enough of the selected photos.");
            co_return;
        }

        std::wstring projectPath = RunProjectPath();

        m_project.CloseProject(); // release any handle from a prior click before deleting the file
        m_storage.Close();
        DeleteFileW(projectPath.c_str());
        DeleteFileW((projectPath + L"-wal").c_str());
        DeleteFileW((projectPath + L"-shm").c_str());

        if (isBurst)
        {
            m_project.CreateBurstProject(projectPath, ToBurstMode(m_selectedFeature));
            // No pre-sized preview canvas - a burst project has no chunk
            // grid and no known output size until BURST_MERGE actually
            // runs (Super Res Zoom upsamples); ShowFinalBurstResult
            // creates m_renderCanvas once BURST_FINISH's real output
            // dimensions are known (see the plan doc's scope decision 2).
        }
        else
        {
            int maxWidth = 0;
            int maxHeight = 0;
            for (const auto& img : pickedImages)
            {
                maxWidth = (std::max)(maxWidth, img.width);
                maxHeight = (std::max)(maxHeight, img.height);
            }

            // Coarse pre-alignment canvas estimate, refined once real
            // homographies exist (Align/Optimize) - not the final extent.
            int totalWidth = maxWidth * static_cast<int>(pickedImages.size());
            int totalHeight = maxHeight * static_cast<int>(pickedImages.size());

            m_project.CreateProject(projectPath, totalWidth, totalHeight);

            // Blank (fully transparent) canvas at the project's full extent -
            // each Render chunk fills in its own sub-rectangle as its task
            // completes (see the onTaskCompleted callback passed to
            // m_pipelineDriver.Initialize below), so the still-transparent
            // regions show the placeholder background through until their
            // chunk lands instead of the whole picture popping in at once.
            //
            // totalWidth/totalHeight is a coarse pre-alignment OVER-estimate
            // (maxWidth/maxHeight * image count, sized for the worst case
            // before real geometry is known) - fine as inert project/chunk-
            // grid bookkeeping, but allocating a WriteableBitmap at that size
            // verbatim is a real crash for real photos (3 x 3000x4000 phone
            // photos -> a 9000x12000 canvas, ~412MB, allocated synchronously
            // on the UI thread with no room to fail gracefully). Cap the
            // *preview* canvas to a sane display size and have
            // BlitRenderedChunk downsample into it instead.
            constexpr int kMaxPreviewDimension = 2048;
            int longestSide = (std::max)(totalWidth, totalHeight);
            m_renderPreviewScale = (longestSide > kMaxPreviewDimension)
                ? static_cast<double>(kMaxPreviewDimension) / static_cast<double>(longestSide)
                : 1.0;
            int previewWidth = (std::max)(1, static_cast<int>(totalWidth * m_renderPreviewScale));
            int previewHeight = (std::max)(1, static_cast<int>(totalHeight * m_renderPreviewScale));

            try
            {
                m_renderCanvas = WriteableBitmap(previewWidth, previewHeight);
                RenderPreviewImage().Source(m_renderCanvas);
                EmptyStatePanel().Visibility(Visibility::Collapsed);
            }
            catch (hresult_error const& error)
            {
                // The live preview is a nice-to-have, not load-bearing - if
                // the canvas still can't be allocated for some other reason,
                // disable it (BlitRenderedChunk no-ops on a null
                // m_renderCanvas) rather than taking the whole app down; the
                // stitch itself doesn't depend on this.
                m_renderCanvas = nullptr;
                AppendToStitchLog(L"Render preview disabled: " + std::wstring(error.message().c_str()));
            }
        }

        {
            size_t lastSlash = projectPath.find_last_of(L'\\');
            size_t lastDot = projectPath.find_last_of(L'.');
            std::wstring projectDirectory = (lastSlash == std::wstring::npos) ? L"." : projectPath.substr(0, lastSlash);
            size_t nameStart = (lastSlash == std::wstring::npos) ? 0 : lastSlash + 1;
            size_t nameLength = (lastDot == std::wstring::npos || lastDot < nameStart) ? std::wstring::npos : lastDot - nameStart;
            std::wstring projectBaseName = projectPath.substr(nameStart, nameLength);

            m_storage.Open(projectDirectory, projectBaseName, m_project);
        }

        for (const auto& img : pickedImages)
        {
            m_project.AddInputImage(img.path, Homography{}, img.cfaType);
        }

        if (isBurst)
        {
            m_project.SeedBurstAlignTasks();
            m_project.SeedBurstMergeTasks();

            m_pipelineDriver.RegisterExecutor(PipelineStage::BURST_ALIGN,
                std::make_shared<::WindowsApp::Core::BurstAlignExecutor>(m_project, m_storage, m_cudaPipeline, m_nvJpegCodec));
            auto mergeExecutor = std::make_shared<::WindowsApp::Core::BurstMergeExecutor>(m_project, m_storage, m_cudaPipeline);
            m_pipelineDriver.RegisterExecutor(PipelineStage::BURST_MERGE, mergeExecutor);
            m_pipelineDriver.RegisterExecutor(PipelineStage::BURST_FINISH, mergeExecutor);
        }
        else
        {
            m_project.SeedIngestTasks();
            m_project.SeedAlignTasks();
            m_project.SeedOptimizeTasks();
            // STAGE3_RENDER is deliberately not seeded here - it needs
            // Optimize's final homographies; PipelineDriver::Run seeds it
            // right before the Render stage (issue #42).

            m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE0_INGEST,
                std::make_shared<::WindowsApp::Core::RawIngestExecutor>(m_project, m_storage, m_cudaPipeline, m_nvJpegCodec));
            m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE1_ALIGN,
                std::make_shared<::WindowsApp::Core::AlignExecutor>(m_project, m_storage, m_cudaPipeline, m_nvJpegCodec));
            m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE2_OPTIMIZE,
                std::make_shared<::WindowsApp::Core::OptimizeExecutor>(m_project, m_storage, m_cudaPipeline, m_nvJpegCodec));
            m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE3_RENDER,
                std::make_shared<::WindowsApp::Core::RenderExecutor>(m_project, m_storage, m_cudaPipeline));
        }

        RunStatusText().Text(L"Starting...");

        m_runStopSource = std::stop_source();
        auto token = m_runStopSource.get_token();
        auto dispatcher = DispatcherQueue();

        // weak_ref, not `this`, in every dispatcher-posted closure below:
        // TryEnqueue only *queues* work, it doesn't wait for it to run, so
        // a closure posted right before the background thread function
        // returns can still be sitting in the queue after m_runThread's
        // destructor (join) has unblocked ~MainWindow() and later members
        // start tearing down. Capturing `this`/members directly is safe
        // everywhere else in this method (and inside Run itself) because
        // that code only ever executes while the background thread - which
        // the destructor joins before those members die - is still alive.
        auto weakThis{ get_weak() };

        m_pipelineDriver.Initialize(
            [weakThis, dispatcher](::WindowsApp::Core::PipelineStage stage, float progress)
            {
                dispatcher.TryEnqueue([weakThis, stage, progress]
                {
                    if (auto strongThis{ weakThis.get() })
                    {
                        strongThis->RunStatusText().Text(StageToDisplayString(stage));
                        strongThis->RunProgressBar().Value(progress * 100.0);
                    }
                });
            },
            [weakThis, dispatcher](std::wstring const& message)
            {
                AppendToStitchLog(message);
                dispatcher.TryEnqueue([weakThis, message]
                {
                    if (auto strongThis{ weakThis.get() })
                    {
                        strongThis->m_lastLogMessage = message;
                        strongThis->RunStatusText().Text(winrt::hstring{ message });
                    }
                });
            },
            m_computeMaxInFlight,
            [weakThis, dispatcher](::WindowsApp::Core::PipelineStage stage, ::WindowsApp::Core::Task const& task)
            {
                // Runs on m_runThread (see PipelineDriver::TaskCallback's
                // doc comment) - only Render chunks (Stitch) and the
                // BURST_FINISH task (burst modes) have anything worth
                // showing.
                if (!task.outputBlobId.has_value()) return;

                auto strongThis{ weakThis.get() };
                if (!strongThis) return;

                if (stage == ::WindowsApp::Core::PipelineStage::STAGE3_RENDER)
                {
                    auto pixels = strongThis->m_storage.ReadPixelBuffer(task.outputBlobId.value());
                    if (!pixels.has_value()) return;

                    int xOffset = 0, yOffset = 0;
                    bool foundChunk = false;
                    for (const auto& chunk : strongThis->m_project.GetChunks())
                    {
                        if (chunk.id == task.unitKey)
                        {
                            xOffset = chunk.x_offset;
                            yOffset = chunk.y_offset;
                            foundChunk = true;
                            break;
                        }
                    }
                    if (!foundChunk) return;

                    // Format conversion (RGB48 -> BGRA8) happens here, off the
                    // UI thread, since it's the expensive part - only the
                    // actual WriteableBitmap buffer write + Invalidate (cheap)
                    // needs to run on the UI thread. Wrapped in a shared_ptr
                    // so the DispatcherQueueHandler delegate stays copyable.
                    auto pixelsPtr = std::make_shared<::WindowsApp::Core::PixelBuffer>(std::move(pixels.value()));
                    dispatcher.TryEnqueue([weakThis, pixelsPtr, xOffset, yOffset]
                    {
                        if (auto strongThis2{ weakThis.get() })
                        {
                            strongThis2->BlitRenderedChunk(*pixelsPtr, xOffset, yOffset);
                        }
                    });
                }
                else if (stage == ::WindowsApp::Core::PipelineStage::BURST_FINISH)
                {
                    auto pixels = strongThis->m_storage.ReadPixelBuffer(task.outputBlobId.value());
                    if (!pixels.has_value()) return;

                    auto pixelsPtr = std::make_shared<::WindowsApp::Core::PixelBuffer>(std::move(pixels.value()));
                    dispatcher.TryEnqueue([weakThis, pixelsPtr]
                    {
                        if (auto strongThis2{ weakThis.get() })
                        {
                            strongThis2->ShowFinalBurstResult(*pixelsPtr);
                        }
                    });
                }
            });

        m_runThread = std::jthread([this, weakThis, dispatcher, token]()
        {
            bool ok = false;
            std::wstring crashMessage;
            // This runs on a raw std::jthread, entirely outside XAML's
            // dispatch stack - App::UnhandledException never sees an
            // exception thrown here, so without this catch it would hit
            // std::terminate() and kill the process with zero log output,
            // not even the crash-dialog/temp-log path other failures get.
            try
            {
                ok = m_pipelineDriver.Run(m_project, token);
            }
            catch (hresult_error const& error)
            {
                crashMessage = L"Run thread threw: " + std::wstring(error.message().c_str());
            }
            catch (std::exception const& error)
            {
                std::string narrow = error.what();
                crashMessage = L"Run thread threw: " + std::wstring(narrow.begin(), narrow.end());
            }
            catch (...)
            {
                crashMessage = L"Run thread threw an unrecognized exception.";
            }
            if (!crashMessage.empty())
            {
                AppendToStitchLog(crashMessage);
            }
            bool wasCancelled = token.stop_requested();

            dispatcher.TryEnqueue([weakThis, ok, wasCancelled, crashMessage]
            {
                if (auto strongThis{ weakThis.get() })
                {
                    strongThis->RunStartButton().IsEnabled(true);
                    strongThis->RunCancelButton().IsEnabled(false);
                    strongThis->ExportButton().IsEnabled(ok);

                    if (!crashMessage.empty())
                    {
                        strongThis->m_lastLogMessage = crashMessage;
                    }

                    winrt::hstring statusText;
                    if (ok)
                    {
                        statusText = L"Run completed.";
                    }
                    else if (wasCancelled)
                    {
                        statusText = L"Run cancelled.";
                    }
                    else if (!strongThis->m_lastLogMessage.empty())
                    {
                        statusText = winrt::hstring{ L"Run failed: " } +
                            winrt::hstring{ strongThis->m_lastLogMessage };
                    }
                    else
                    {
                        statusText = L"Run failed. See %TEMP%\\WindowsApp.log for details.";
                    }
                    strongThis->RunStatusText().Text(statusText);

                    if (ok)
                    {
                        strongThis->RunProgressBar().Value(100.0);
                    }
                }
            });
        });
    }

    void MainWindow::RunCancelButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        // request_stop() only - never join here, that would block the UI
        // thread. The background thread's own completion handler (already
        // dispatched back to the UI thread once Run actually returns)
        // re-enables Start.
        m_runStopSource.request_stop();
        RunStatusText().Text(L"Cancelling - finishing in-flight work...");
    }

    void MainWindow::BlitRenderedChunk(const ::WindowsApp::Core::PixelBuffer& pixels, int xOffset, int yOffset)
    {
        if (!m_renderCanvas) return;

        int canvasWidth = m_renderCanvas.PixelWidth();
        int canvasHeight = m_renderCanvas.PixelHeight();

        // IBuffer::data() (a C++/WinRT convenience extension, not a
        // projected WinRT member) hands back the raw backing pointer
        // directly - avoids the classic IBufferByteAccess/robuffer.h COM
        // interop dance, which also isn't safe to include here anyway
        // (its raw ::Windows namespace collides with this file's
        // `using namespace winrt;` + `using namespace Windows::Storage;`).
        auto buffer = m_renderCanvas.PixelBuffer();
        BYTE* canvasBytes = buffer.data();

        const int canvasStride = canvasWidth * 4; // BGRA8, tightly packed

        // Destination rectangle for this chunk in the (possibly
        // downsampled - see m_renderPreviewScale's declaration)
        // preview canvas. Point-sampling (nearest-neighbor) the source
        // for each destination pixel, not the other way around, so a
        // downscale never leaves gaps - fine for a live progress
        // preview, no need for real filtering.
        int dstX0 = static_cast<int>(xOffset * m_renderPreviewScale);
        int dstY0 = static_cast<int>(yOffset * m_renderPreviewScale);
        int dstWidth = (std::max)(1, static_cast<int>(pixels.width * m_renderPreviewScale));
        int dstHeight = (std::max)(1, static_cast<int>(pixels.height * m_renderPreviewScale));

        for (int dy = 0; dy < dstHeight; ++dy)
        {
            int canvasY = dstY0 + dy;
            if (canvasY < 0 || canvasY >= canvasHeight) continue;

            int srcY = (std::min)(pixels.height - 1, static_cast<int>(dy / m_renderPreviewScale));
            const unsigned short* srcRow = pixels.data.data() + static_cast<size_t>(srcY) * pixels.width * 3;
            BYTE* dstRow = canvasBytes + static_cast<size_t>(canvasY) * canvasStride;

            for (int dx = 0; dx < dstWidth; ++dx)
            {
                int canvasX = dstX0 + dx;
                if (canvasX < 0 || canvasX >= canvasWidth) continue;

                int srcX = (std::min)(pixels.width - 1, static_cast<int>(dx / m_renderPreviewScale));

                unsigned short r = srcRow[static_cast<size_t>(srcX) * 3 + 0];
                unsigned short g = srcRow[static_cast<size_t>(srcX) * 3 + 1];
                unsigned short b = srcRow[static_cast<size_t>(srcX) * 3 + 2];

                BYTE* dstPixel = dstRow + static_cast<size_t>(canvasX) * 4;
                dstPixel[0] = static_cast<BYTE>(b >> 8);
                dstPixel[1] = static_cast<BYTE>(g >> 8);
                dstPixel[2] = static_cast<BYTE>(r >> 8);
                dstPixel[3] = 0xFF;
            }
        }

        m_renderCanvas.Invalidate();
    }

    void MainWindow::ShowFinalBurstResult(const ::WindowsApp::Core::PixelBuffer& pixels)
    {
        // Same kMaxPreviewDimension downsample policy as Stitch's canvas
        // pre-allocation (see RunStartButton_Click) - a Super Res Zoom
        // result can be several thousand pixels per side, and a
        // WriteableBitmap at that size verbatim risks the same real
        // crash Stitch's own comment describes.
        constexpr int kMaxPreviewDimension = 2048;
        int longestSide = (std::max)(pixels.width, pixels.height);
        m_renderPreviewScale = (longestSide > kMaxPreviewDimension)
            ? static_cast<double>(kMaxPreviewDimension) / static_cast<double>(longestSide)
            : 1.0;
        int previewWidth = (std::max)(1, static_cast<int>(pixels.width * m_renderPreviewScale));
        int previewHeight = (std::max)(1, static_cast<int>(pixels.height * m_renderPreviewScale));

        try
        {
            m_renderCanvas = WriteableBitmap(previewWidth, previewHeight);
            RenderPreviewImage().Source(m_renderCanvas);
            EmptyStatePanel().Visibility(Visibility::Collapsed);
        }
        catch (hresult_error const& error)
        {
            m_renderCanvas = nullptr;
            AppendToStitchLog(L"Result preview disabled: " + std::wstring(error.message().c_str()));
            return;
        }

        BlitRenderedChunk(pixels, 0, 0);
    }

    void MainWindow::ComputeBackendComboBox_SelectionChanged(
        Windows::Foundation::IInspectable const& /* sender */,
        Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& /* args */)
    {
        switch (ComputeBackendComboBox().SelectedIndex())
        {
        case 1: m_computeBackendPreference = ::WindowsApp::ComputeBackendKind::Cuda; break;
        case 2: m_computeBackendPreference = ::WindowsApp::ComputeBackendKind::Vulkan; break;
        case 3: m_computeBackendPreference = ::WindowsApp::ComputeBackendKind::Cpu; break;
        default: m_computeBackendPreference = ::WindowsApp::ComputeBackendKind::Auto; break;
        }

        // The already-initialized backend (if any) was picked under the
        // previous preference - force SelectComputeBackend() to run again
        // with the new one on the next Run click instead of silently
        // keeping whatever backend the first run happened to pick.
        // Ignored for burst modes (forced CPU regardless - see
        // RunStartButton_Click), but harmless to set unconditionally.
        m_computeInitialized = false;
    }

    winrt::fire_and_forget MainWindow::ExportButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        auto lifetime{ get_strong() }; // keep `this` alive across co_await suspension

        if (m_exportThread.joinable())
        {
            m_exportThread.join();
        }

        bool isBurst = IsBurstFeature(m_selectedFeature);

        FileSavePicker picker{ AppWindow().Id() };
        if (isBurst)
        {
            // TIFF (lossless, via Tiff16Writer - the same writer
            // image360_cli's burst subcommands use) and JPEG (lossy, via
            // the existing JpegCodec path) - PanoramaExporter's
            // JPEG-only ExportPreviewJpeg doesn't apply here, burst
            // output has no chunk grid to composite (plan doc scope
            // decision 3).
            picker.SuggestedFileName(L"result");
            picker.FileTypeChoices().Insert(L"TIFF image (16-bit)", winrt::single_threaded_vector<winrt::hstring>({ L".tif" }));
            picker.FileTypeChoices().Insert(L"JPEG image", winrt::single_threaded_vector<winrt::hstring>({ L".jpg" }));
        }
        else
        {
            picker.SuggestedFileName(L"panorama_preview");
            picker.FileTypeChoices().Insert(L"JPEG image", winrt::single_threaded_vector<winrt::hstring>({ L".jpg" }));
        }

        auto destFile{ co_await picker.PickSaveFileAsync() };
        if (!destFile)
        {
            RunStatusText().Text(L"Export cancelled.");
            co_return;
        }

        std::wstring destPath(destFile.Path().c_str());

        ExportButton().IsEnabled(false);
        RunStatusText().Text(L"Exporting...");

        auto weakThis{ get_weak() }; // same reasoning as RunStartButton_Click - see that method's comment
        auto dispatcher = DispatcherQueue();

        if (isBurst)
        {
            m_exportThread = std::jthread([this, weakThis, dispatcher, destPath]()
            {
                bool ok = false;

                std::vector<::WindowsApp::Core::Task> finishTasks =
                    m_project.GetTasksForStage(::WindowsApp::Core::PipelineStage::BURST_FINISH);
                if (finishTasks.size() == 1 && finishTasks[0].status == ::WindowsApp::Core::TaskStatus::COMPLETED
                    && finishTasks[0].outputBlobId.has_value())
                {
                    auto pixels = m_storage.ReadPixelBuffer(*finishTasks[0].outputBlobId);
                    if (pixels.has_value())
                    {
                        auto endsWithCaseInsensitive = [&](const wchar_t* suffix)
                        {
                            std::wstring suffixStr(suffix);
                            if (destPath.size() < suffixStr.size()) return false;
                            std::wstring tail = destPath.substr(destPath.size() - suffixStr.size());
                            std::transform(tail.begin(), tail.end(), tail.begin(), ::towlower);
                            return tail == suffixStr;
                        };

                        if (endsWithCaseInsensitive(L".tif") || endsWithCaseInsensitive(L".tiff"))
                        {
                            ok = ::WindowsApp::Core::WriteTiff16RGB(destPath, pixels->data.data(), pixels->width, pixels->height);
                        }
                        else
                        {
                            std::vector<unsigned char> rgb8(pixels->data.size());
                            for (size_t i = 0; i < pixels->data.size(); ++i)
                                rgb8[i] = static_cast<unsigned char>(pixels->data[i] >> 8);

                            std::vector<unsigned char> jpegBytes;
                            if (m_nvJpegCodec->Encode(rgb8.data(), pixels->width, pixels->height, 90, jpegBytes)
                                == ::WindowsApp::Compute::ComputeResult::SUCCESS)
                            {
                                ::WindowsApp::Core::PlatformFile file;
                                if (file.Open(destPath, ::WindowsApp::Core::FileOpenMode::CreateAlways))
                                {
                                    ok = file.Write(jpegBytes.data(), jpegBytes.size());
                                    file.Close();
                                }
                            }
                        }
                    }
                }

                dispatcher.TryEnqueue([weakThis, ok]
                {
                    if (auto strongThis{ weakThis.get() })
                    {
                        strongThis->ExportButton().IsEnabled(true);
                        strongThis->RunStatusText().Text(ok ? L"Result exported." : L"Export failed.");
                    }
                });
            });
        }
        else
        {
            m_exportThread = std::jthread([this, weakThis, dispatcher, destPath]()
            {
                ::WindowsApp::Core::PanoramaExporter exporter(m_project, m_storage, m_nvJpegCodec);
                bool ok = exporter.ExportPreviewJpeg(destPath, 4096);

                dispatcher.TryEnqueue([weakThis, ok]
                {
                    if (auto strongThis{ weakThis.get() })
                    {
                        strongThis->ExportButton().IsEnabled(true);
                        strongThis->RunStatusText().Text(
                            ok ? L"Preview JPEG exported." : L"Preview export failed.");
                    }
                });
            });
        }
    }
}
