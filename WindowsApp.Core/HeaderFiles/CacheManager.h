#pragma once
#include "Types.h"
#include <string>

namespace WindowsApp::Core
{
    class CacheManager
    {
    public:
        void WriteChunkToDisk(const std::wstring& filePath, const PixelBuffer& buffer);
        bool ReadChunkFromDisk(const std::wstring& filePath, PixelBuffer& buffer);
    };
}
