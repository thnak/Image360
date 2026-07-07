#include "pch.h"
#include "MainWindow.xaml.h"
#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif

#include <chrono>
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
    // Deliberately minimal stand-in for real GPU work - RawIngest/Align/
    // Optimize/Render executors don't exist yet (see this plan's Global
    // Constraints in docs/superpowers/plans/2026-07-07-ui-progress-nonblocking.md).
    // Lives here, not WindowsApp.Tests, because the test-only
    // StubTaskExecutor must not be linked into the shipping UI binary.
    class DemoStitchExecutor : public ::WindowsApp::Core::ITaskExecutor
    {
    public:
        bool Execute(::WindowsApp::Core::Task& /* task */, ::WindowsApp::Core::CancellationToken /* token */) override
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            return true;
        }
    };

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

    void MainWindow::StitchStartButton_Click(
        Windows::Foundation::IInspectable const& /* sender */,
        RoutedEventArgs const& /* args */)
    {
        using ::WindowsApp::Core::PipelineStage;
        using ::WindowsApp::Core::Task;

        StitchStartButton().IsEnabled(false);
        StitchCancelButton().IsEnabled(true);
        StitchStatusText().Text(L"Starting stitch...");
        StitchProgressBar().Value(0);

        // A previous run has already finished (Start is disabled while one
        // is active), but std::jthread stays joinable until explicitly
        // joined - clear it out before starting a fresh one.
        if (m_stitchThread.joinable())
        {
            m_stitchThread.join();
        }

        winrt::hstring localFolder = ApplicationData::Current().LocalFolder().Path();
        std::wstring projectPath = std::wstring(localFolder.c_str()) + L"\\demo_stitch.vfp";

        // Deliberately minimal stand-in for real project creation (see this
        // plan's Global Constraints) - CreateProject is safe to call again
        // on an existing file: its schema uses CREATE TABLE IF NOT EXISTS
        // and never touches existing task rows.
        m_stitchProject.CreateProject(projectPath, 8192, 8192);

        if (m_stitchProject.GetTasksForStage(PipelineStage::STAGE1_ALIGN).empty())
        {
            std::vector<Task> seeds;
            for (int i = 0; i < 4; ++i)
            {
                Task t;
                t.stage = PipelineStage::STAGE1_ALIGN;
                t.unitKind = "image";
                t.unitKey = "demo_img_" + std::to_string(i);
                seeds.push_back(t);
            }
            m_stitchProject.CreateTasksIfAbsent(seeds);
        }

        m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE1_ALIGN, std::make_shared<DemoStitchExecutor>());

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
                dispatcher.TryEnqueue([weakThis, message]
                {
                    if (auto strongThis{ weakThis.get() })
                    {
                        strongThis->StitchStatusText().Text(winrt::hstring{ message });
                    }
                });
            });

        m_stitchThread = std::jthread([this, weakThis, dispatcher, token]()
        {
            bool ok = m_pipelineDriver.Run(m_stitchProject, token);
            bool wasCancelled = token.stop_requested();

            dispatcher.TryEnqueue([weakThis, ok, wasCancelled]
            {
                if (auto strongThis{ weakThis.get() })
                {
                    strongThis->StitchStartButton().IsEnabled(true);
                    strongThis->StitchCancelButton().IsEnabled(false);
                    strongThis->StitchStatusText().Text(
                        ok ? L"Stitch completed." : (wasCancelled ? L"Stitch cancelled." : L"Stitch failed."));
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
}
