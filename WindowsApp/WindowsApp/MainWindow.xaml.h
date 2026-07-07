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

        winrt::fire_and_forget OpenImageButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        void ResetEditsButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        void AdjustmentSlider_ValueChanged(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::Controls::Primitives::RangeBaseValueChangedEventArgs const& args);

        winrt::fire_and_forget StitchStartButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

        void StitchCancelButton_Click(
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

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
        std::shared_ptr<::WindowsApp::Compute::CudaPipeline> m_cudaPipeline;
        std::shared_ptr<::WindowsApp::Compute::NvJpegCodec> m_nvJpegCodec;
        bool m_computeInitialized = false;
        std::stop_source m_stitchStopSource;
        std::jthread m_stitchThread;
    };
}

namespace winrt::WindowsApp::factory_implementation
{
    struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow>
    {
    };
}
