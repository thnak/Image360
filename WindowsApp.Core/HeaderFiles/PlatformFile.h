#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace WindowsApp::Core
{
    enum class FileOpenMode
    {
        ReadOnly,     // must already exist
        OpenOrCreate, // read+write; create if missing, keep existing content, offset 0
        CreateAlways, // write-only; create fresh, truncating any existing content
        CreateNew     // read+write; fails if the file already exists
    };

    // Cross-platform binary file handle backing StorageEngine's sharded
    // blob files and PanoramaExporter's final export. Windows impl wraps
    // CreateFileW/ReadFile/WriteFile; POSIX impl wraps fopen/fread/fwrite.
    // Reads/writes are chunked internally to match the existing on-disk
    // I/O granularity.
    class PlatformFile
    {
    public:
        PlatformFile() = default;
        ~PlatformFile();
        PlatformFile(const PlatformFile&) = delete;
        PlatformFile& operator=(const PlatformFile&) = delete;

        bool Open(const std::wstring& path, FileOpenMode mode);
        void Close();
        bool IsOpen() const;

        bool SeekAbsolute(uint64_t offset);
        uint64_t Size() const;

        bool Write(const void* data, size_t length);
        bool Read(void* data, size_t length);

        static bool Exists(const std::wstring& path);

    private:
        void* m_handle = nullptr;
        bool m_isOpen = false;
    };
}
