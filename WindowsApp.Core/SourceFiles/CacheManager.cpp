#include "pch.h"
#include "HeaderFiles/CacheManager.h"
#include <stdexcept>

namespace WindowsApp::Core
{
    void CacheManager::WriteChunkToDisk(const std::wstring& filePath, const PixelBuffer& buffer)
    {
        size_t dataSize = buffer.data.size() * sizeof(unsigned short);

        HANDLE hFile = CreateFileW(filePath.c_str(), GENERIC_READ | GENERIC_WRITE,
            0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Failed to create cache file.");

        HANDLE hMap = CreateFileMappingW(hFile, NULL, PAGE_READWRITE,
            0, static_cast<DWORD>(dataSize), NULL);
        if (!hMap)
        {
            CloseHandle(hFile);
            throw std::runtime_error("Failed to create file mapping.");
        }

        void* pData = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, dataSize);
        if (pData)
        {
            memcpy(pData, buffer.data.data(), dataSize);
            UnmapViewOfFile(pData);
        }

        CloseHandle(hMap);
        CloseHandle(hFile);
    }

    bool CacheManager::ReadChunkFromDisk(const std::wstring& filePath, PixelBuffer& buffer)
    {
        HANDLE hFile = CreateFileW(filePath.c_str(), GENERIC_READ,
            FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile, &fileSize))
        {
            CloseHandle(hFile);
            return false;
        }

        size_t dataSize = static_cast<size_t>(fileSize.QuadPart);
        HANDLE hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap)
        {
            CloseHandle(hFile);
            return false;
        }

        const void* pData = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, dataSize);
        if (pData)
        {
            size_t numPixels = dataSize / sizeof(unsigned short);
            buffer.data.resize(numPixels);
            memcpy(buffer.data.data(), pData, dataSize);
            UnmapViewOfFile(pData);
        }

        CloseHandle(hMap);
        CloseHandle(hFile);
        return pData != nullptr;
    }
}
