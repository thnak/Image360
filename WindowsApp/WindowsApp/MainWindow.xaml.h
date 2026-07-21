#pragma once

#include "MainWindow.g.h"

namespace winrt::WindowsApp::implementation
{
    struct MainWindow : MainWindowT<MainWindow>
    {
        MainWindow()
        {
            // Xaml objects should not call InitializeComponent during construction.
            // See https://github.com/microsoft/cppwinrt/tree/master/nuget#initializecomponent
        }

        int32_t MyProperty();
        void MyProperty(int32_t value);

        winrt::fire_and_forget StitchStartButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        void ComputeBackendComboBox_SelectionChanged(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& args);

        void StitchCancelButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        winrt::fire_and_forget StitchExportButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        // Must run on the UI thread (touches m_renderCanvas/RenderPreviewImage) -
        // the StitchStartButton_Click's per-task callback dispatches into
        // this via DispatcherQueue::TryEnqueue, same as every other UI
        // touch from that background thread. Blits `pixels` (RGB48) into
        // m_renderCanvas at (xOffset, yOffset), converting to the
        // WriteableBitmap's native BGRA8 format, and invalidates so the
        // compositor picks up the change - this is what makes each render
        // chunk appear as soon as its task completes instead of only the
        // final assembled panorama.
        void BlitRenderedChunk(const ::WindowsApp::Core::PixelBuffer& pixels, int xOffset, int yOffset);

    private:
        // Declared in this order - not the stop_source/jthread/Project/
        // Driver order the plan doc sketches - because member destruction
        // runs in REVERSE declaration order. m_stitchThread must be
        // destroyed (std::jthread's destructor: request_stop() + join())
        // before m_pipelineDriver/m_stitchProject are destroyed, or a
        // still-running background thread would call into already-freed
        // objects during MainWindow teardown. This is what makes
        // window-close-mid-run safe with no extra shutdown handler.
        ::WindowsApp::Core::ProjectManager m_stitchProject;
        ::WindowsApp::Core::StorageEngine m_stitchStorage;
        ::WindowsApp::Core::PipelineDriver m_pipelineDriver;
        // Sized to the project's full canvas (totalWidth x totalHeight) at
        // Start Stitch time and set as RenderPreviewImage's Source; each
        // completed Render chunk is blitted into its (x_offset, y_offset)
        // sub-rectangle via BlitRenderedChunk as the stitch progresses.
        // Only ever touched from the UI thread.
        winrt::Microsoft::UI::Xaml::Media::Imaging::WriteableBitmap m_renderCanvas{ nullptr };
        // 1.0 unless the project's full canvas exceeds a sane preview
        // size (see StitchStartButton_Click) - real multi-photo panoramas
        // can have a coarse pre-alignment canvas estimate of several
        // thousand pixels per side per input image, and allocating a
        // WriteableBitmap at that size verbatim is hundreds of MB to
        // multiple GB (a real crash seen with 3000x4000 phone photos).
        // BlitRenderedChunk downsamples each chunk's pixels by this factor
        // instead of copying 1:1.
        double m_renderPreviewScale = 1.0;
        // Concrete type is CudaPipeline or CpuComputeBackend, picked at
        // runtime by ComputeBackendFactory::SelectComputeBackend() - see
        // its call site for why (no GPU should abort the whole app).
        std::shared_ptr<::WindowsApp::Compute::IComputeBackend> m_cudaPipeline;
        std::shared_ptr<::WindowsApp::Compute::IImageCodec> m_nvJpegCodec;
        bool m_computeInitialized = false;
        // Set from ComputeBackendComboBox; changing it invalidates
        // m_computeInitialized so the next Start Stitch click re-runs
        // SelectComputeBackend() with the new preference instead of
        // reusing whatever backend the first run happened to pick.
        ::WindowsApp::ComputeBackendKind m_computeBackendPreference = ::WindowsApp::ComputeBackendKind::Auto;
        size_t m_computeMaxInFlight = 2;
        // Last message PipelineDriver's onLog callback delivered - the
        // stitch-completion handler folds this into "Stitch failed." since
        // the raw log line only ever lives in StitchStatusText for a
        // moment before being overwritten. Only ever touched from the UI
        // thread (both writers are DispatcherQueue::TryEnqueue closures).
        std::wstring m_lastStitchLogMessage;
        std::stop_source m_stitchStopSource;
        std::jthread m_stitchThread;
        // Declared last for the same reverse-destruction-order reason as
        // m_stitchThread - joined (via std::jthread's destructor) before
        // any member it might still be using during MainWindow teardown.
        std::jthread m_exportThread;
    };
}

namespace winrt::WindowsApp::factory_implementation
{
    struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow>
    {
    };
}
