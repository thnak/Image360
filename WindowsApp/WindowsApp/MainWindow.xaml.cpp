#include "pch.h"
#include "MainWindow.xaml.h"
#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif

using namespace winrt;
using namespace Microsoft::UI::Xaml;
using namespace Microsoft::UI::Xaml::Controls::Primitives;
using namespace Microsoft::UI::Xaml::Media::Imaging;
using namespace Microsoft::Windows::Storage::Pickers;
using namespace Windows::Storage;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

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
}
