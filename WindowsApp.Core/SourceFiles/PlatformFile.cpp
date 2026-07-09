#include "pch.h"
#include "HeaderFiles/PlatformFile.h"
#include "HeaderFiles/TextEncoding.h"

#include <algorithm>

#ifndef _WIN32
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#endif

namespace WindowsApp::Core
{
    namespace
    {
        constexpr size_t kIoChunkBytes = 64u * 1024 * 1024; // 64 MB per read/write call
    }

    PlatformFile::~PlatformFile()
    {
        Close();
    }

    bool PlatformFile::IsOpen() const
    {
        return m_isOpen;
    }

#ifdef _WIN32
    bool PlatformFile::Open(const std::wstring& path, FileOpenMode mode)
    {
        Close();

        DWORD access = 0;
        DWORD shareMode = FILE_SHARE_READ;
        DWORD disposition = OPEN_EXISTING;
        switch (mode)
        {
        case FileOpenMode::ReadOnly:
            access = GENERIC_READ;
            shareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
            disposition = OPEN_EXISTING;
            break;
        case FileOpenMode::OpenOrCreate:
            access = GENERIC_READ | GENERIC_WRITE;
            disposition = OPEN_ALWAYS;
            break;
        case FileOpenMode::CreateAlways:
            access = GENERIC_WRITE;
            shareMode = 0;
            disposition = CREATE_ALWAYS;
            break;
        case FileOpenMode::CreateNew:
            access = GENERIC_READ | GENERIC_WRITE;
            disposition = CREATE_NEW;
            break;
        }

        HANDLE handle = CreateFileW(path.c_str(), access, shareMode, nullptr, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) return false;

        m_handle = reinterpret_cast<void*>(handle);
        m_isOpen = true;
        return true;
    }

    void PlatformFile::Close()
    {
        if (m_isOpen)
        {
            CloseHandle(reinterpret_cast<HANDLE>(m_handle));
            m_handle = nullptr;
            m_isOpen = false;
        }
    }

    bool PlatformFile::SeekAbsolute(uint64_t offset)
    {
        if (!m_isOpen) return false;
        LARGE_INTEGER seekPos;
        seekPos.QuadPart = static_cast<LONGLONG>(offset);
        return SetFilePointerEx(reinterpret_cast<HANDLE>(m_handle), seekPos, nullptr, FILE_BEGIN) != 0;
    }

    uint64_t PlatformFile::Size() const
    {
        if (!m_isOpen) return 0;
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(reinterpret_cast<HANDLE>(m_handle), &size)) return 0;
        return static_cast<uint64_t>(size.QuadPart);
    }

    bool PlatformFile::Write(const void* data, size_t length)
    {
        if (!m_isOpen) return false;
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        size_t totalWritten = 0;
        while (totalWritten < length)
        {
            DWORD toWrite = static_cast<DWORD>(std::min<size_t>(length - totalWritten, kIoChunkBytes));
            DWORD written = 0;
            if (!WriteFile(reinterpret_cast<HANDLE>(m_handle), bytes + totalWritten, toWrite, &written, nullptr) || written == 0)
                return false;
            totalWritten += written;
        }
        return true;
    }

    bool PlatformFile::Read(void* data, size_t length)
    {
        if (!m_isOpen) return false;
        uint8_t* bytes = static_cast<uint8_t*>(data);
        size_t totalRead = 0;
        while (totalRead < length)
        {
            DWORD toRead = static_cast<DWORD>(std::min<size_t>(length - totalRead, kIoChunkBytes));
            DWORD readBytes = 0;
            if (!ReadFile(reinterpret_cast<HANDLE>(m_handle), bytes + totalRead, toRead, &readBytes, nullptr) || readBytes == 0)
                return false;
            totalRead += readBytes;
        }
        return true;
    }

    bool PlatformFile::Exists(const std::wstring& path)
    {
        return GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES;
    }
#else
    namespace
    {
        int ToFd(void* handle)
        {
            return static_cast<int>(reinterpret_cast<std::intptr_t>(handle));
        }
    }

    bool PlatformFile::Open(const std::wstring& path, FileOpenMode mode)
    {
        Close();

        std::string utf8Path = WideToUtf8(path);
        int flags = 0;
        mode_t perms = 0644;
        switch (mode)
        {
        case FileOpenMode::ReadOnly:
            flags = O_RDONLY;
            break;
        case FileOpenMode::OpenOrCreate:
            flags = O_RDWR | O_CREAT;
            break;
        case FileOpenMode::CreateAlways:
            flags = O_WRONLY | O_CREAT | O_TRUNC;
            break;
        case FileOpenMode::CreateNew:
            flags = O_RDWR | O_CREAT | O_EXCL;
            break;
        }

        int fd = ::open(utf8Path.c_str(), flags, perms);
        if (fd < 0) return false;

        m_handle = reinterpret_cast<void*>(static_cast<std::intptr_t>(fd));
        m_isOpen = true;
        return true;
    }

    void PlatformFile::Close()
    {
        if (m_isOpen)
        {
            ::close(ToFd(m_handle));
            m_handle = nullptr;
            m_isOpen = false;
        }
    }

    bool PlatformFile::SeekAbsolute(uint64_t offset)
    {
        if (!m_isOpen) return false;
        return ::lseek(ToFd(m_handle), static_cast<off_t>(offset), SEEK_SET) != static_cast<off_t>(-1);
    }

    uint64_t PlatformFile::Size() const
    {
        if (!m_isOpen) return 0;
        struct stat st{};
        if (::fstat(ToFd(m_handle), &st) != 0) return 0;
        return static_cast<uint64_t>(st.st_size);
    }

    bool PlatformFile::Write(const void* data, size_t length)
    {
        if (!m_isOpen) return false;
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        size_t totalWritten = 0;
        while (totalWritten < length)
        {
            size_t toWrite = std::min<size_t>(length - totalWritten, kIoChunkBytes);
            ssize_t written = ::write(ToFd(m_handle), bytes + totalWritten, toWrite);
            if (written <= 0) return false;
            totalWritten += static_cast<size_t>(written);
        }
        return true;
    }

    bool PlatformFile::Read(void* data, size_t length)
    {
        if (!m_isOpen) return false;
        uint8_t* bytes = static_cast<uint8_t*>(data);
        size_t totalRead = 0;
        while (totalRead < length)
        {
            size_t toRead = std::min<size_t>(length - totalRead, kIoChunkBytes);
            ssize_t readBytes = ::read(ToFd(m_handle), bytes + totalRead, toRead);
            if (readBytes <= 0) return false;
            totalRead += static_cast<size_t>(readBytes);
        }
        return true;
    }

    bool PlatformFile::Exists(const std::wstring& path)
    {
        std::string utf8Path = WideToUtf8(path);
        return ::access(utf8Path.c_str(), F_OK) == 0;
    }
#endif
}
