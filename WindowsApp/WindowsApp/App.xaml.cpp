#include "pch.h"
#include "App.xaml.h"
#include "MainWindow.xaml.h"
#include <fstream>
#include <string>
#include <ctime>

using namespace winrt;
using namespace Microsoft::UI::Xaml;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace winrt::WindowsApp::implementation
{
    /// <summary>
    /// Initializes the singleton application object.  This is the first line of authored code
    /// executed, and as such is the logical equivalent of main() or WinMain().
    /// </summary>
    App::App()
    {
        // Xaml objects should not call InitializeComponent during construction.
        // See https://github.com/microsoft/cppwinrt/tree/master/nuget#initializecomponent

        UnhandledException([](IInspectable const&, UnhandledExceptionEventArgs const& e)
        {
            auto errorMessage = e.Message();

            // Log exception to a local file in Temp
            wchar_t tempPath[MAX_PATH];
            if (GetTempPathW(MAX_PATH, tempPath))
            {
                std::wstring logFilePath = std::wstring(tempPath) + L"WindowsApp.log";
                std::wofstream logFile(logFilePath, std::ios::app);
                if (logFile.is_open())
                {
                    time_t now = time(nullptr);
                    wchar_t timeStr[26];
                    _wctime_s(timeStr, 26, &now);
                    std::wstring timeString(timeStr);
                    if (!timeString.empty() && timeString.back() == L'\n')
                    {
                        timeString.pop_back();
                    }
                    logFile << L"[" << timeString << L"] Unhandled Exception: " << errorMessage.c_str() << std::endl;
                }
            }

            // Pop up a message box so the crash isn't silent
            MessageBoxW(nullptr, errorMessage.c_str(), L"WindowsApp - Fatal Error", MB_OK | MB_ICONERROR);

            if (IsDebuggerPresent())
            {
                __debugbreak();
            }
        });
    }

    /// <summary>
    /// Invoked when the application is launched.
    /// </summary>
    /// <param name="e">Details about the launch request and process.</param>
    void App::OnLaunched([[maybe_unused]] LaunchActivatedEventArgs const& e)
    {
        window = make<MainWindow>();
        window.Activate();
    }
}
