#include "pch.h"
#include "MainWindow.xaml.h"
#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif

#include "ImageLoader.h"
#include "PanoramaExporter.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

using namespace winrt;
using namespace Microsoft::UI::Xaml;
using namespace Microsoft::UI::Xaml::Controls::Primitives;
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
        case PipelineStage::COMPLETED:       return L"Stitch completed.";
        case PipelineStage::CANCELLED:       return L"Stitch cancelled.";
        case PipelineStage::FAILED:          return L"Stitch failed.";
        }
        return L"Stitching...";
    }

    // ApplicationData::Current() throws ("The process has no package
    // identity") unless the app is running with MSIX package identity -
    // i.e. launched via the WindowsApp (Package) project, not the bare
    // WindowsApp.exe. Fall back to %LOCALAPPDATA%\Image360 so the demo
    // stitch works either way instead of crashing when run unpackaged.
    std::wstring StitchProjectPath()
    {
        try
        {
            winrt::hstring localFolder = ApplicationData::Current().LocalFolder().Path();
            return std::wstring(localFolder.c_str()) + L"\\demo_stitch.vfp";
        }
        catch (hresult_error const&)
        {
            wchar_t buffer[MAX_PATH];
            DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", buffer, MAX_PATH);
            std::wstring folder = (len > 0 && len < MAX_PATH) ? std::wstring(buffer) : L".";
            folder += L"\\Image360";
            CreateDirectoryW(folder.c_str(), nullptr);
            return folder + L"\\demo_stitch.vfp";
        }
    }

    // Same file/format App::App()'s UnhandledException handler writes to -
    // one place to check regardless of whether a run failed gracefully
    // (this) or crashed outright (that). Pipeline log lines only ever
    // reach the UI as a transient StitchStatusText string that the
    // completion handler immediately overwrites with "Stitch failed.", so
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

    fire_and_forget MainWindow::OpenImageButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        auto lifetime{ get_strong() };

        try
        {
            FileOpenPicker picker{ AppWindow().Id() };
            picker.FileTypeFilter().Append(L".jpg");
            picker.FileTypeFilter().Append(L".jpeg");
            picker.FileTypeFilter().Append(L".png");
            picker.FileTypeFilter().Append(L".bmp");
            picker.FileTypeFilter().Append(L".gif");
            picker.FileTypeFilter().Append(L".tif");
            picker.FileTypeFilter().Append(L".tiff");

            auto result{ co_await picker.PickSingleFileAsync() };
            if (!result)
            {
                EditorStatusText().Text(L"Picture selection cancelled.");
                co_return;
            }

            auto file{ co_await StorageFile::GetFileFromPathAsync(result.Path()) };
            auto stream{ co_await file.OpenReadAsync() };

            BitmapImage bitmap;
            co_await bitmap.SetSourceAsync(stream);

            EditedImage().Source(bitmap);
            EmptyStatePanel().Visibility(Visibility::Collapsed);
            EditorStatusText().Text(L"Picture loaded. Adjustments are staged for the GPU preview pipeline.");
        }
        catch (hresult_error const& error)
        {
            EditorStatusText().Text(winrt::hstring{ L"Could not open picture: " } + error.message());
        }
    }

    void MainWindow::ResetEditsButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        BrightnessSlider().Value(0);
        ContrastSlider().Value(0);
        SaturationSlider().Value(0);
        SharpnessSlider().Value(0);
        EffectModeComboBox().SelectedIndex(0);
        GpuPreviewToggle().IsOn(true);
        EditorStatusText().Text(L"Edits reset to the original preview.");
    }

    void MainWindow::AdjustmentSlider_ValueChanged(
        Windows::Foundation::IInspectable const& /* sender */,
        RangeBaseValueChangedEventArgs const& /* args */)
    {
        if (auto statusText{ EditorStatusText() })
        {
            statusText.Text(L"Adjustment changed. GPU preview backend is the next step for rendered effects.");
        }
    }

    winrt::fire_and_forget MainWindow::StitchStartButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        using ::WindowsApp::Core::CfaType;
        using ::WindowsApp::Core::Homography;
        using ::WindowsApp::Core::ImageMetadata;
        using ::WindowsApp::Core::PipelineStage;
        using ::WindowsApp::Core::RawPlane;
        using ::WindowsApp::Core::Task;

        auto lifetime{ get_strong() }; // keep `this` alive across co_await suspension

        StitchStartButton().IsEnabled(false);
        StitchCancelButton().IsEnabled(true);
        StitchStatusText().Text(L"Pick RAW photos to stitch...");
        StitchProgressBar().Value(0);
        m_lastStitchLogMessage.clear();

        // A previous run has already finished (Start is disabled while one
        // is active), but std::jthread stays joinable until explicitly
        // joined - clear it out before starting a fresh one.
        if (m_stitchThread.joinable())
        {
            m_stitchThread.join();
        }

        // Lazily initialize the engine objects once per session (not per
        // run) - avoids paying CUDA init cost if the user never clicks
        // Start Stitch, and this is the first place in the shipping UI a
        // missing/incompatible GPU becomes a real, user-visible failure
        // mode rather than a demo that never touched CUDA at all.
        if (!m_computeInitialized)
        {
            m_cudaPipeline = std::make_shared<::WindowsApp::Compute::CudaPipeline>();
            if (m_cudaPipeline->Initialize() != ::WindowsApp::Compute::ComputeResult::SUCCESS)
            {
                const char* errorMsg = m_cudaPipeline->GetLastError();
                std::wstring wideError(errorMsg, errorMsg + strlen(errorMsg)); // ASCII-only error text

                AppendToStitchLog(L"No compatible GPU found: " + wideError);
                StitchStartButton().IsEnabled(true);
                StitchCancelButton().IsEnabled(false);
                StitchStatusText().Text(winrt::hstring{ L"No compatible GPU found: " } + winrt::hstring{ wideError });
                co_return;
            }

            m_nvJpegCodec = std::make_shared<::WindowsApp::Compute::NvJpegCodec>();
            if (m_nvJpegCodec->Initialize() != ::WindowsApp::Compute::ComputeResult::SUCCESS)
            {
                AppendToStitchLog(L"Failed to initialize the JPEG codec.");
                StitchStartButton().IsEnabled(true);
                StitchCancelButton().IsEnabled(false);
                StitchStatusText().Text(L"Failed to initialize the JPEG codec.");
                co_return;
            }

            m_computeInitialized = true;
        }

        FileOpenPicker picker{ AppWindow().Id() };
        picker.FileTypeFilter().Append(L".raf");
        picker.FileTypeFilter().Append(L".cr2");
        picker.FileTypeFilter().Append(L".nef");
        picker.FileTypeFilter().Append(L".arw");
        picker.FileTypeFilter().Append(L".dng");

        auto pickedFiles{ co_await picker.PickMultipleFilesAsync() };
        if (!pickedFiles || pickedFiles.Size() == 0)
        {
            StitchStartButton().IsEnabled(true);
            StitchCancelButton().IsEnabled(false);
            StitchStatusText().Text(L"No photos selected.");
            co_return;
        }

        StitchStatusText().Text(L"Reading photo metadata...");

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

        if (pickedImages.empty())
        {
            StitchStartButton().IsEnabled(true);
            StitchCancelButton().IsEnabled(false);
            StitchStatusText().Text(L"Could not read any of the selected photos.");
            co_return;
        }

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

        std::wstring projectPath = StitchProjectPath();

        m_stitchProject.CloseProject(); // release any handle from a prior click before deleting the file
        m_stitchStorage.Close();
        DeleteFileW(projectPath.c_str());
        DeleteFileW((projectPath + L"-wal").c_str());
        DeleteFileW((projectPath + L"-shm").c_str());
        m_stitchProject.CreateProject(projectPath, totalWidth, totalHeight);

        {
            size_t lastSlash = projectPath.find_last_of(L'\\');
            size_t lastDot = projectPath.find_last_of(L'.');
            std::wstring projectDirectory = (lastSlash == std::wstring::npos) ? L"." : projectPath.substr(0, lastSlash);
            size_t nameStart = (lastSlash == std::wstring::npos) ? 0 : lastSlash + 1;
            size_t nameLength = (lastDot == std::wstring::npos || lastDot < nameStart) ? std::wstring::npos : lastDot - nameStart;
            std::wstring projectBaseName = projectPath.substr(nameStart, nameLength);

            m_stitchStorage.Open(projectDirectory, projectBaseName, m_stitchProject);
        }

        for (const auto& img : pickedImages)
        {
            m_stitchProject.AddInputImage(img.path, Homography{}, img.cfaType);
        }

        m_stitchProject.SeedIngestTasks();
        m_stitchProject.SeedAlignTasks();
        m_stitchProject.SeedOptimizeTasks();
        // STAGE3_RENDER is deliberately not seeded here - it needs
        // Optimize's final homographies; PipelineDriver::Run seeds it
        // right before the Render stage (issue #42).

        m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE0_INGEST,
            std::make_shared<::WindowsApp::Core::RawIngestExecutor>(m_stitchProject, m_stitchStorage, m_cudaPipeline));
        m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE1_ALIGN,
            std::make_shared<::WindowsApp::Core::AlignExecutor>(m_stitchProject, m_stitchStorage, m_cudaPipeline, m_nvJpegCodec));
        m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE2_OPTIMIZE,
            std::make_shared<::WindowsApp::Core::OptimizeExecutor>(m_stitchProject, m_stitchStorage, m_cudaPipeline, m_nvJpegCodec));
        m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE3_RENDER,
            std::make_shared<::WindowsApp::Core::RenderExecutor>(m_stitchProject, m_stitchStorage, m_cudaPipeline));

        StitchStatusText().Text(L"Starting stitch...");

        m_stitchStopSource = std::stop_source();
        auto token = m_stitchStopSource.get_token();
        auto dispatcher = DispatcherQueue();

        // weak_ref, not `this`, in every dispatcher-posted closure below:
        // TryEnqueue only *queues* work, it doesn't wait for it to run, so
        // a closure posted right before the background thread function
        // returns can still be sitting in the queue after m_stitchThread's
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
                        strongThis->StitchStatusText().Text(StageToDisplayString(stage));
                        strongThis->StitchProgressBar().Value(progress * 100.0);
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
                        strongThis->m_lastStitchLogMessage = message;
                        strongThis->StitchStatusText().Text(winrt::hstring{ message });
                    }
                });
            });

        m_stitchThread = std::jthread([this, weakThis, dispatcher, token]()
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
                ok = m_pipelineDriver.Run(m_stitchProject, token);
            }
            catch (hresult_error const& error)
            {
                crashMessage = L"Stitch thread threw: " + std::wstring(error.message().c_str());
            }
            catch (std::exception const& error)
            {
                std::string narrow = error.what();
                crashMessage = L"Stitch thread threw: " + std::wstring(narrow.begin(), narrow.end());
            }
            catch (...)
            {
                crashMessage = L"Stitch thread threw an unrecognized exception.";
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
                    strongThis->StitchStartButton().IsEnabled(true);
                    strongThis->StitchCancelButton().IsEnabled(false);
                    strongThis->StitchExportButton().IsEnabled(ok);

                    if (!crashMessage.empty())
                    {
                        strongThis->m_lastStitchLogMessage = crashMessage;
                    }

                    winrt::hstring statusText;
                    if (ok)
                    {
                        statusText = L"Stitch completed.";
                    }
                    else if (wasCancelled)
                    {
                        statusText = L"Stitch cancelled.";
                    }
                    else if (!strongThis->m_lastStitchLogMessage.empty())
                    {
                        statusText = winrt::hstring{ L"Stitch failed: " } +
                            winrt::hstring{ strongThis->m_lastStitchLogMessage };
                    }
                    else
                    {
                        statusText = L"Stitch failed. See %TEMP%\\WindowsApp.log for details.";
                    }
                    strongThis->StitchStatusText().Text(statusText);

                    if (ok)
                    {
                        strongThis->StitchProgressBar().Value(100.0);
                    }
                }
            });
        });
    }

    void MainWindow::StitchCancelButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        // request_stop() only - never join here, that would block the UI
        // thread. The background thread's own completion handler (already
        // dispatched back to the UI thread once Run actually returns)
        // re-enables Start.
        m_stitchStopSource.request_stop();
        StitchStatusText().Text(L"Cancelling - finishing in-flight work...");
    }

    winrt::fire_and_forget MainWindow::StitchExportButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        auto lifetime{ get_strong() }; // keep `this` alive across co_await suspension

        if (m_exportThread.joinable())
        {
            m_exportThread.join();
        }

        FileSavePicker picker{ AppWindow().Id() };
        picker.SuggestedFileName(L"panorama_preview");
        picker.FileTypeChoices().Insert(L"JPEG image", winrt::single_threaded_vector<winrt::hstring>({ L".jpg" }));

        auto destFile{ co_await picker.PickSaveFileAsync() };
        if (!destFile)
        {
            StitchStatusText().Text(L"Export cancelled.");
            co_return;
        }

        std::wstring destPath(destFile.Path().c_str());

        StitchExportButton().IsEnabled(false);
        StitchStatusText().Text(L"Exporting preview JPEG...");

        auto weakThis{ get_weak() }; // same reasoning as StitchStartButton_Click - see that method's comment
        auto dispatcher = DispatcherQueue();

        m_exportThread = std::jthread([this, weakThis, dispatcher, destPath]()
        {
            ::WindowsApp::Core::PanoramaExporter exporter(m_stitchProject, m_stitchStorage, m_nvJpegCodec);
            bool ok = exporter.ExportPreviewJpeg(destPath, 4096);

            dispatcher.TryEnqueue([weakThis, ok]
            {
                if (auto strongThis{ weakThis.get() })
                {
                    strongThis->StitchExportButton().IsEnabled(true);
                    strongThis->StitchStatusText().Text(
                        ok ? L"Preview JPEG exported." : L"Preview export failed.");
                }
            });
        });
    }
}
