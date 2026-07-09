#include "pch.h"
#include "HeaderFiles/TextEncoding.h"

#include <cstdint>

namespace WindowsApp::Core
{
#ifdef _WIN32
    std::wstring Utf8ToWide(const std::string& utf8)
    {
        if (utf8.empty()) return std::wstring();
        int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
        std::wstring wide(wlen - 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, wide.data(), wlen);
        return wide;
    }

    std::string WideToUtf8(const std::wstring& wide)
    {
        if (wide.empty()) return std::string();
        int len = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string utf8(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, utf8.data(), len, nullptr, nullptr);
        return utf8;
    }
#else
    namespace
    {
        void AppendUtf8(std::string& out, char32_t cp)
        {
            if (cp <= 0x7F)
            {
                out.push_back(static_cast<char>(cp));
            }
            else if (cp <= 0x7FF)
            {
                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            else if (cp <= 0xFFFF)
            {
                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            else
            {
                out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        }
    }

    std::wstring Utf8ToWide(const std::string& utf8)
    {
        std::wstring result;
        result.reserve(utf8.size());
        size_t i = 0;
        while (i < utf8.size())
        {
            unsigned char c = static_cast<unsigned char>(utf8[i]);
            char32_t cp = 0;
            size_t extra = 0;
            if (c <= 0x7F) { cp = c; extra = 0; }
            else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
            else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
            else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
            else { ++i; continue; }

            if (i + extra >= utf8.size()) break;
            bool valid = true;
            for (size_t k = 1; k <= extra; ++k)
            {
                unsigned char cc = static_cast<unsigned char>(utf8[i + k]);
                if ((cc & 0xC0) != 0x80) { valid = false; break; }
                cp = (cp << 6) | (cc & 0x3F);
            }
            if (!valid) { ++i; continue; }
            result.push_back(static_cast<wchar_t>(cp));
            i += extra + 1;
        }
        return result;
    }

    std::string WideToUtf8(const std::wstring& wide)
    {
        std::string result;
        result.reserve(wide.size());
        for (wchar_t wc : wide)
        {
            AppendUtf8(result, static_cast<char32_t>(static_cast<uint32_t>(wc)));
        }
        return result;
    }
#endif
}
