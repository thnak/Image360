#pragma once

#include "MainWindow.g.h"

namespace winrt::WindowsApp::implementation
{
    // Which pipeline Run/Export drive - selected via FeatureComboBox.
    // Stitch is the default (index 0), preserving this app's original
    // panorama-only behavior for anyone who never touches the combo.
    // docs/superpowers/plans/2026-07-22-winui-feature-management.md.
    enum class AppFeature
    {
        Stitch,
        Mfnr,
        HdrPlus,
        NightSight,
        SuperRes,
    };

    struct MainWindow : MainWindowT<MainWindow>
    {
        MainWindow()
        {
            // Xaml objects should not call InitializeComponent during construction.
            // See https://github.com/microsoft/cppwinrt/tree/master/nuget#initializecomponent
        }

        int32_t MyProperty();
        void MyProperty(int32_t value);

        winrt::fire_and_forget RunStartButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        void FeatureComboBox_SelectionChanged(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& args);

        void ComputeBackendComboBox_SelectionChanged(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& args);

        void RunCancelButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        winrt::fire_and_forget ExportButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        // Must run on the UI thread (touches m_renderCanvas/RenderPreviewImage) -
        // the RunStartButton_Click's per-task callback dispatches into
        // this via DispatcherQueue::TryEnqueue, same as every other UI
        // touch from that background thread. Blits `pixels` (RGB48) into
        // m_renderCanvas at (xOffset, yOffset), converting to the
        // WriteableBitmap's native BGRA8 format, and invalidates so the
        // compositor picks up the change - this is what makes each render
        // chunk appear as soon as its task completes instead of only the
        // final assembled panorama. Stitch only (STAGE3_RENDER) - the
        // canvas is pre-sized before this is ever called.
        void BlitRenderedChunk(const ::WindowsApp::Core::PixelBuffer& pixels, int xOffset, int yOffset);

        // Burst modes' single-shot equivalent of BlitRenderedChunk - a
        // burst project has no chunk grid and no known output size until
        // BURST_MERGE actually runs (Super Res Zoom upsamples), so unlike
        // Stitch the preview canvas can't be pre-sized; this creates it at
        // `pixels`' real dimensions and blits the whole result in one
        // shot (docs/superpowers/plans/2026-07-22-winui-feature-management.md
        // scope decision 2). Must run on the UI thread, same as
        // BlitRenderedChunk.
        void ShowFinalBurstResult(const ::WindowsApp::Core::PixelBuffer& pixels);

    private:
        // Declared in this order - not the stop_source/jthread/Project/
        // Driver order the plan doc sketches - because member destruction
        // runs in REVERSE declaration order. m_runThread must be
        // destroyed (std::jthread's destructor: request_stop() + join())
        // before m_pipelineDriver/m_project are destroyed, or a
        // still-running background thread would call into already-freed
        // objects during MainWindow teardown. This is what makes
        // window-close-mid-run safe with no extra shutdown handler.
        ::WindowsApp::Core::ProjectManager m_project;
        ::WindowsApp::Core::StorageEngine m_storage;
        ::WindowsApp::Core::PipelineDriver m_pipelineDriver;
        // Which pipeline the current/last Run targets - set from
        // FeatureComboBox, defaults to Stitch (see AppFeature's comment).
        AppFeature m_selectedFeature = AppFeature::Stitch;
        // Sized to the project's full canvas (totalWidth x totalHeight) at
        // Run time and set as RenderPreviewImage's Source; each
        // completed Render chunk is blitted into its (x_offset, y_offset)
        // sub-rectangle via BlitRenderedChunk as the stitch progresses
        // (Stitch only - burst modes create/replace this once, in
        // ShowFinalBurstResult). Only ever touched from the UI thread.
        winrt::Microsoft::UI::Xaml::Media::Imaging::WriteableBitmap m_renderCanvas{ nullptr };
        // 1.0 unless the project's full canvas exceeds a sane preview
        // size (see RunStartButton_Click) - real multi-photo panoramas
        // can have a coarse pre-alignment canvas estimate of several
        // thousand pixels per side per input image, and allocating a
        // WriteableBitmap at that size verbatim is hundreds of MB to
        // multiple GB (a real crash seen with 3000x4000 phone photos).
        // BlitRenderedChunk/ShowFinalBurstResult downsample by this
        // factor instead of copying 1:1.
        double m_renderPreviewScale = 1.0;
        // Concrete type is CudaPipeline/VulkanPipeline/CpuComputeBackend,
        // picked at runtime by ComputeBackendFactory::SelectComputeBackend()
        // for Stitch, or forced to CpuComputeBackend for burst modes (see
        // m_lastBackendForcedForBurst below and the plan doc's scope
        // decision 1 - burst-mode kernels have no Vulkan/CUDA
        // implementation yet).
        std::shared_ptr<::WindowsApp::Compute::IComputeBackend> m_cudaPipeline;
        std::shared_ptr<::WindowsApp::Compute::IImageCodec> m_nvJpegCodec;
        bool m_computeInitialized = false;
        // Whether m_cudaPipeline/m_nvJpegCodec were selected under the
        // forced-CPU burst-mode policy (true) or a real
        // ComputeBackendKind preference (false) - a feature switch
        // between Stitch and a burst mode (or vice versa) needs the
        // opposite policy, so RunStartButton_Click compares this against
        // the newly selected feature to decide whether to re-select
        // rather than reuse the cached backend.
        bool m_lastBackendForcedForBurst = false;
        // Set from ComputeBackendComboBox; changing it invalidates
        // m_computeInitialized so the next Run re-runs
        // SelectComputeBackend() with the new preference instead of
        // reusing whatever backend the first run happened to pick.
        // Ignored entirely when the selected feature is a burst mode
        // (forced CPU regardless).
        ::WindowsApp::ComputeBackendKind m_computeBackendPreference = ::WindowsApp::ComputeBackendKind::Auto;
        size_t m_computeMaxInFlight = 2;
        // Last message PipelineDriver's onLog callback delivered - the
        // run-completion handler folds this into "Run failed." since
        // the raw log line only ever lives in RunStatusText for a
        // moment before being overwritten. Only ever touched from the UI
        // thread (both writers are DispatcherQueue::TryEnqueue closures).
        std::wstring m_lastLogMessage;
        std::stop_source m_runStopSource;
        std::jthread m_runThread;
        // Declared last for the same reverse-destruction-order reason as
        // m_runThread - joined (via std::jthread's destructor) before
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
