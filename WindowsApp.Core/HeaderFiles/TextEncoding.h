#pragma once

#include <string>

namespace WindowsApp::Core
{
    // UTF-8 <-> platform-native wide string conversion. On Windows, wchar_t
    // is UTF-16 and this wraps WideCharToMultiByte/MultiByteToWideChar; on
    // other platforms wchar_t is UTF-32 and this does a direct codepoint
    // conversion. Used at LibRaw/SQLite boundaries, which want UTF-8 char*.
    std::wstring Utf8ToWide(const std::string& utf8);
    std::string WideToUtf8(const std::wstring& wide);
}
